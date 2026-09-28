import io
import json
import re
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field
import praw
from openai import OpenAI
import streamlit as st

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Reddit AI Pain Point Finder (DeepSeek)",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# ESQUEMA DE DATOS
# ==========================================
class DiagnosticoProblemaIA(BaseModel):
    es_solucionable_con_ia: bool = Field(
        description="True si el cuello de botella o problema operativo puede resolverse de forma viable con inteligencia artificial."
    )
    rol_o_industria: str = Field(
        default="No especificado",
        description="Cargo, profesión o tipo de negocio deducible del autor."
    )
    resumen_queja: str = Field(
        description="Resumen conciso y claro de la queja, cuello de botella o fricción operativa."
    )
    solucion_ia_propuesta: str = Field(
        description="Explicación concreta de cómo la IA resuelve este problema."
    )
    stack_tecnologico_sugerido: str = Field(
        description="Herramientas o modelos recomendados (ej. DeepSeek API, Whisper, LangChain, Make, RAG)."
    )

# ==========================================
# FUNCIÓN DE ANÁLISIS VÍA DEEPSEEK
# ==========================================
def analizar_post_con_deepseek(client: OpenAI, modelo: str, titulo: str, cuerpo: str, autor: str) -> Optional[DiagnosticoProblemaIA]:
    prompt = f"""
    Analiza la siguiente publicación de Reddit:
    AUTOR: {autor}
    TÍTULO: {titulo}
    CONTENIDO: {cuerpo}

    Responde ÚNICAMENTE con un objeto JSON válido con los siguientes campos exactos:
    {{
        "es_solucionable_con_ia": boolean,
        "rol_o_industria": string,
        "resumen_queja": string,
        "solucion_ia_propuesta": string,
        "stack_tecnologico_sugerido": string
    }}

    Criterios:
    1. 'es_solucionable_con_ia' debe ser true únicamente si la queja o fricción se resuelve mediante modelos de lenguaje, visión, agentes, extracción documental o predicción.
    2. Si no es solucionable con IA, devuelve 'es_solucionable_con_ia': false.
    """
    try:
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Eres un auditor senior de procesos de negocio e ingeniería de soluciones de IA. Devuelves estrictamente formato JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        return DiagnosticoProblemaIA(**data)
    except Exception as e:
        st.error(f"Error durante el análisis con DeepSeek: {e}")
        return None

# ==========================================
# INTERFAZ DE USUARIO
# ==========================================
st.title("⚡ Detector de Oportunidades de IA en Reddit (Vía DeepSeek)")
st.markdown(
    "Monitorea subreddits en busca de quejas y fricciones operativas, evalúa con **DeepSeek** "
    "su viabilidad técnica y genera automáticamente una hoja Excel al validar **10 resultados**."
)

with st.sidebar:
    st.header("🔑 Configuración de DeepSeek")
    deepseek_api_key = st.text_input("DeepSeek API Key", type="password", help="Obtén tu clave en platform.deepseek.com")
    modelo_seleccionado = st.text_input("Identificador de Modelo", value="deepseek-chat", help="Usa el alias o ID del modelo rápido de DeepSeek (ej. deepseek-chat o deepseek-flash)")

    st.subheader("Reddit API (PRAW)")
    reddit_client_id = st.text_input("Reddit Client ID", type="password")
    reddit_client_secret = st.text_input("Reddit Client Secret", type="password")
    reddit_user_agent = st.text_input("User Agent", value="DeepSeekPainMiner/1.0 by Streamlit")

    st.markdown("---")
    st.caption("La API de DeepSeek utiliza la URL base oficial `https://api.deepseek.com`.")

# Entradas principales
col1, col2 = st.columns([1, 1])

with col1:
    subreddits_input = st.text_area(
        "Subreddits a rastrear (separados por comas o saltos de línea):",
        value="smallbusiness, entrepreneur, sysadmin, productivity, operations",
        height=100
    )

with col2:
    keywords_input = st.text_area(
        "Frases o síntomas de dolor operativo:",
        value="tired of doing manually\ntakes too much time\nhate doing this\nbottleneck in process\nwasting hours on",
        height=100
    )

iniciar_escaneo = st.button("🚀 Iniciar Rastreo y Extracción (10 resultados)", type="primary")

# ==========================================
# PROCESAMIENTO Y MONITOREO
# ==========================================
if iniciar_escaneo:
    if not (deepseek_api_key and reddit_client_id and reddit_client_secret):
        st.warning("⚠️ Debes proporcionar tu clave de DeepSeek y las credenciales de Reddit.")
        st.stop()

    lista_subreddits = [s.strip().replace("r/", "") for s in re.split(r"[,\n]+", subreddits_input) if s.strip()]
    lista_queries = [q.strip() for q in re.split(r"[\n]+", keywords_input) if q.strip()]

    if not lista_subreddits:
        st.error("Ingresa al menos un subreddit válido.")
        st.stop()

    # Inicialización de clientes
    try:
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # Conexión directa a la API de DeepSeek
        deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
    except Exception as err:
        st.error(f"Error al inicializar servicios: {err}")
        st.stop()

    subreddits_str = "+".join(lista_subreddits)
    resultados = []
    ids_visitados = set()
    limite_objetivo = 10

    barra_progreso = st.progress(0, text="Iniciando conexión...")
    contenedor_estado = st.empty()
    tabla_en_vivo = st.empty()

    for query in lista_queries:
        if len(resultados) >= limite_objetivo:
            break

        contenedor_estado.info(f"Buscando: **'{query}'** en `r/{subreddits_str}`...")
        
        try:
            posts = reddit.subreddit(subreddits_str).search(query=query, sort="new", limit=30)
            
            for post in posts:
                if len(resultados) >= limite_objetivo:
                    break

                if post.id in ids_visitados or not post.is_self:
                    continue
                ids_visitados.add(post.id)

                texto = f"{post.title}\n{post.selftext}"
                if len(texto.strip()) < 80:
                    continue

                autor_nom = post.author.name if post.author else "[Anonimo]"
                
                # Evaluación vía DeepSeek
                evaluacion = analizar_post_con_deepseek(
                    deepseek_client, 
                    modelo_seleccionado, 
                    post.title, 
                    post.selftext, 
                    autor_nom
                )

                if evaluacion and evaluacion.es_solucionable_con_ia:
                    resultados.append({
                        "N°": len(resultados) + 1,
                        "Subreddit": f"r/{post.subreddit.display_name}",
                        "Usuario": f"u/{autor_nom}",
                        "URL Publicación": f"https://reddit.com{post.permalink}",
                        "Rol / Industria": evaluacion.rol_o_industria,
                        "Queja / Cuello de Botella": evaluacion.resumen_queja,
                        "Solución con IA Propuesta": evaluacion.solucion_ia_propuesta,
                        "Stack Tecnológico Sugerido": evaluacion.stack_tecnologico_sugerido
                    })

                    pct = len(resultados) / limite_objetivo
                    barra_progreso.progress(pct, text=f"Casos validados con DeepSeek: {len(resultados)} / {limite_objetivo}")
                    
                    df_temporal = pd.DataFrame(resultados)
                    tabla_en_vivo.dataframe(
                        df_temporal[["N°", "Subreddit", "Queja / Cuello de Botella", "Solución con IA Propuesta"]], 
                        use_container_width=True
                    )

        except Exception as e:
            st.warning(f"Incidencia con la búsqueda '{query}': {e}")
            continue

    contenedor_estado.empty()

    # ==========================================
    # DESCARGA DE RESULTADOS EN EXCEL
    # ==========================================
    if len(resultados) > 0:
        st.success(f"🎯 Meta completada: {len(resultados)} problemas viables identificados.")
        df_final = pd.DataFrame(resultados)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_final.to_excel(writer, index=False, sheet_name="Oportunidades IA")
            
            ws = writer.sheets["Oportunidades IA"]
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                col_letter = col[0].column_letter
                ws.column_dimensions[col_letter].width = min(max(max_len + 3, 14), 55)

        buffer.seek(0)

        st.download_button(
            label="📥 Descargar Reporte Excel (.xlsx)",
            data=buffer,
            file_name="reddit_problemas_ia_deepseek.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    else:
        st.warning("No se hallaron publicaciones que cumplieran los criterios en este lote.")
