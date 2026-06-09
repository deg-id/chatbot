# CHATBOT RAG CON DEEPSEEK + FAISS (OPTIMIZADO PARA RENDER GRATUITO)
# ----------------------------------------------------------------
# - Carga múltiples PDFs, TXTs y MDs desde "especialidad"
# - Usa FAISS + Embeddings ultra-ligeros (23MB en RAM)
# - Conexión directa a la API oficial de DeepSeek

import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

# Componentes de LangChain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI  # Usamos el cliente compatible con OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# CONFIGURACIONES
CARPETA_ESPECIALIDAD = Path("especialidad")
FORMATOS_SOPORTADOS = {".pdf": "PDF", ".txt": "TXT", ".md": "Markdown"}

# Modelo ultra-ligero ideal para no reventar los 512MB de Render
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-albert-small-v2"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# VALIDACIÓN DE CREDENCIALES DEEPSEEK
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "❌ ERROR: No se encontró la variable DEEPSEEK_API_KEY.\n"
        "Asegúrate de configurar tu archivo '.env' local o las variables en Render."
    )

# LÓGICA DE CARGA DE DOCUMENTOS
def cargar_individual(ruta_archivo: Path) -> list[Document]:
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

    texto = ruta_archivo.read_text(encoding="utf-8", errors="ignore").strip()
    if texto:
        return [Document(page_content=texto, metadata={"source": ruta_archivo.name})]
    return []

def cargar_todos_los_documentos() -> list[Document]:
    if not CARPETA_ESPECIALIDAD.exists():
        CARPETA_ESPECIALIDAD.mkdir()
        
    documentos_totales = []
    for archivo in CARPETA_ESPECIALIDAD.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in FORMATOS_SOPORTADOS:
            try:
                logger.info(f"Leyendo archivo: {archivo.name}")
                docs = cargar_individual(archivo)
                documentos_totales.extend(docs)
            except Exception as e:
                logger.error(f"Error con {archivo.name}: {e}")
                
    if not documentos_totales:
        logger.warning("Carpeta 'especialidad' vacía. Usando documento de respaldo.")
        documentos_totales.append(
            Document(page_content="Horarios de atención: Lunes a Domingo de 12:00 a 23:00.", metadata={"source": "sistema"})
        )
    return documentos_totales

# CONFIGURACIÓN DEL PROMPT PARA EL RESTAURANTE
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
Eres el asistente virtual experto y corporativo del Restaurante La Orquídea. 
Tu única tarea es responder a la pregunta del usuario utilizando estrictamente el bloque de contexto provisto.

Reglas de oro:
1. Básate únicamente en los datos del contexto. Está prohibido inventar platos o precios.
2. Si la información solicitada no está en el contexto, responde textualmente:
   "Lo siento, por el momento no poseo esa información detallada. Te recomiendo contactar a nuestros canales oficiales mediante WhatsApp al +56 9 8765 4321 o visitar nuestro Sitio Web www.laorquidea.cl"
3. Mantén un tono profesional, acogedor y enfocado al cliente. Usa **negritas** para destacar precios o platos.

Contexto:
{context}

Pregunta del cliente: {question}

Respuesta:""")

def format_docs(docs):
    return "\n\n".join(f"[Fuente: {doc.metadata.get('source', 'Desconocido')}]: {doc.page_content}" for doc in docs)

# PREPARACIÓN DE ELEMENTOS (PROCESAMIENTO RAG)
logger.info("Cargando y fragmentando conocimiento...")
documentos_originales = cargar_todos_los_documentos()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documentos_originales)

# Inicializamos embeddings locales ultraligeros
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# Guardamos en la base de datos vectorial indexada en memoria (FAISS)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Inicializamos DeepSeek apuntando a sus servidores oficiales
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    base_url="https://api.deepseek.com/v1",  # URL de compatibilidad oficial de DeepSeek
    api_key=DEEPSEEK_API_KEY,
    temperature=0.3
)

# Cadena RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)

def respond(message, history):
    try:
        return rag_chain.invoke(message)
    except Exception as e:
        logger.error(f"Error en invocación RAG: {e}")
        return "Lo siento, ocurrió un problema al conectar con la base de conocimiento. Inténtalo de nuevo."

# GRADIO INTERFAZ
demo = gr.ChatInterface(
    fn=respond,
    title="Restaurante La Orquídea - Asistente Virtual (DeepSeek)",
    description="Pregúntame sobre nuestra carta gastronómica, ubicación, horarios de atención y reservas.",
    examples=["¿Cuál es el horario de atención?", "¿Qué platos tienen?", "¿Tienen algún número de contacto?"],
)

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 7860))
    logger.info(f"Lanzando Gradio en el puerto: {puerto}")
    demo.launch(server_name="0.0.0.0", server_port=puerto, theme="soft")
