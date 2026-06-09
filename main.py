# =====================================================================
# CHATBOT RAG CON DEEPSEEK V4 FLASH + FAISS
# =====================================================================
# - Entorno: Optimizado para la capa gratuita de Render.com (Bajo consumo de RAM)
# - Gestor de paquetes: uv (Instalación ultra-rápida)
# - Base de Datos: FAISS en memoria para evitar almacenamiento persistente
# - Variables de entorno: Únicamente requiere 'DEEPSEEK_API_KEY'
# =====================================================================

# SECCIÓN 1: IMPORTS Y CONFIGURACIÓN DE LOGS
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

# Componentes principales de LangChain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI  # Usamos el cliente compatible con la API de DeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Configuración de logs visibles en el dashboard de Render
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Carga de archivo .env (solo para desarrollo local)
load_dotenv()


# SECCIÓN 2: CONSTANTES Y DIRECTORIOS
CARPETA_ESPECIALIDAD = Path("especialidad")
FORMATOS_SOPORTADOS = {".pdf": "PDF", ".txt": "TXT", ".md": "Markdown"}

# Modelo de embeddings ultra-ligero (~23MB en RAM) para no saturar el límite de 512MB de Render
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-albert-small-v2"


# SECCIÓN 3: VALIDACIÓN ESTRICTA DE CREDENCIALES
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "❌ ERROR CRÍTICO: No se encontró la variable de entorno 'DEEPSEEK_API_KEY'.\n"
        "Por favor, configúrala en el panel de Environment de Render.com antes de arrancar."
    )


# SECCIÓN 4: EXTRACCIÓN Y CARGA DE DOCUMENTOS (PDF / TXT / MD)
def cargar_individual(ruta_archivo: Path) -> list[Document]:
    """Lee un archivo individual según su formato y lo transforma en documentos de LangChain."""
    if ruta_archivo.suffix.lower() == ".pdf":
        reader = PdfReader(str(ruta_archivo))
        docs_pdf = []
        for i, page in enumerate(reader.pages):
            texto = page.extract_text() or ""
            texto = texto.strip()
            if texto:
                docs_pdf.append(
                    Document(
                        page_content=texto,
                        metadata={"source": ruta_archivo.name, "page": i + 1},
                    )
                )
        return docs_pdf

    # Procesamiento para archivos planos TXT y Markdown
    texto = ruta_archivo.read_text(encoding="utf-8", errors="ignore").strip()
    if texto:
        return [Document(page_content=texto, metadata={"source": ruta_archivo.name})]
    return []


def cargar_todos_los_documentos() -> list[Document]:
    """Escanea la carpeta 'especialidad' y unifica toda la base de conocimiento."""
    if not CARPETA_ESPECIALIDAD.exists():
        CARPETA_ESPECIALIDAD.mkdir()
        logger.info(f"Se creó la carpeta vacía '{CARPETA_ESPECIALIDAD}'.")
        
    documentos_totales = []
    for archivo in CARPETA_ESPECIALIDAD.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in FORMATOS_SOPORTADOS:
            try:
                logger.info(f"Cargando conocimiento desde: {archivo.name}")
                docs = cargar_individual(archivo)
                documentos_totales.extend(docs)
            except Exception as e:
                logger.error(f"Error procesando el archivo {archivo.name}: {e}")
                
    # Documento de emergencia en caso de que olvides subir archivos a GitHub
    if not documentos_totales:
        logger.warning("La carpeta 'especialidad' no contiene archivos válidos. Usando respaldo del sistema.")
        documentos_totales.append(
            Document(page_content="Horarios de atención: Lunes a Domingo de 12:00 a 23:00.", metadata={"source": "sistema"})
        )
    return documentos_totales


# SECCIÓN 5: CONFIGURACIÓN DEL PROMPT DEL ASISTENTE
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
Eres el asistente virtual experto y corporativo del Restaurante La Orquídea. 
Tu única tarea es responder a la pregunta del usuario utilizando estrictamente el bloque de contexto provisto.

Reglas de oro:
1. Básate únicamente en los datos del contexto. Está estrictamente prohibido inventar platos, ingredientes o precios.
2. Si la información solicitada no se encuentra en el contexto, responde textualmente:
   "Lo siento, por el momento no poseo esa información detallada. Te recomiendo contactar a nuestros canales oficiales mediante WhatsApp al +56 9 8765 4321 o visitar nuestro Sitio Web www.laorquidea.cl"
3. Mantén un tono profesional, acogedor y enfocado al cliente. Usa **negritas** para destacar datos clave.

Contexto:
{context}

Pregunta del cliente: {question}

Respuesta:""")


def format_docs(docs):
    """Formatea los fragmentos recuperados para inyectarlos limpiamente en el prompt."""
    return "\n\n".join(f"[Fuente: {doc.metadata.get('source')}]: {doc.page_content}" for doc in docs)


# SECCIÓN 6: PROCESAMIENTO RAG E INICIALIZACIÓN EN MEMORIA
logger.info("=== Iniciando procesamiento RAG ===")
documentos_originales = cargar_todos_los_documentos()

# Fragmentación del texto en bloques manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documentos_originales)

# Descarga del modelo de embeddings ligero de HuggingFace en memoria local
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# Generación de la base vectorial FAISS (se almacena de forma volátil en RAM)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Configuración del LLM conectado a los servidores oficiales de DeepSeek
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",  
    api_key=DEEPSEEK_API_KEY,
    temperature=0.3
)

# Estructura de la cadena LangChain (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)


# SECCIÓN 7: FUNCIÓN DE RESPUESTA DE GRADIO
def respond(message, history):
    """Función puente para la interfaz de Gradio."""
    try:
        return rag_chain.invoke(message)
    except Exception as e:
        logger.error(f"Error en la ejecución de la cadena RAG: {e}")
        return "Lo siento, ocurrió un inconveniente al conectar con mi base de conocimientos. Por favor, intenta de nuevo."


# SECCIÓN 8: INTERFAZ GRÁFICA DE USUARIO
demo = gr.ChatInterface(
    fn=respond,
    title="Restaurante La Orquídea - Asistente Virtual",
    description="Bienvenido. Pregúntame sobre nuestra oferta gastronómica, horarios, reservas o ubicación.",
    examples=["¿Cuál es el horario de atención?", "¿Qué platos tienen?", "¿Tienen algún número de contacto?"],
)


# SECCIÓN 9: BLOQUE DE EJECUCIÓN (ESPECÍFICO PARA RENDER)
if __name__ == "__main__":
    # Render asigna el puerto mediante la variable de entorno 'PORT'. Si no existe, usa el 7860 local.
    puerto = int(os.environ.get("PORT", 7860))
    logger.info(f"Gradio listo para escuchar en el puerto asignado por la plataforma: {puerto}")
    
    # Lanzamiento del servidor web expuesto
    demo.launch(
        server_name="0.0.0.0", 
        server_port=puerto, 
        theme="soft"
    )
