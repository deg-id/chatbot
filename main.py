import logging
import os
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

# Imports ligeros (No consumen RAM ni tiempo al iniciar)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

CARPETA_ESPECIALIDAD = Path("especialidad")
FORMATOS_SOPORTADOS = {".pdf": "PDF", ".txt": "TXT", ".md": "Markdown"}
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("❌ ERROR: Configura 'DEEPSEEK_API_KEY' en Render.")

# --- EL RESTO DE TUS FUNCIONES DE CARGA DE ARCHIVOS SE QUEDAN IGUAL ---
def cargar_individual(ruta_archivo: Path) -> list[Document]:
    # ... (tu código actual de cargar_individual)
    return []

def cargar_todos_los_documentos() -> list[Document]:
    # ... (tu código actual de cargar_todos_los_documentos)
    return []

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""...""") # Tu prompt igual

def format_docs(docs):
    return "\n\n".join(f"[Fuente: {doc.metadata.get('source')}]: {doc.page_content}" for doc in docs)

# =====================================================================
# MODIFICACIÓN CRÍTICA: INICIALIZACIÓN EN SEGUNDO PLANO TOTAL
# =====================================================================
rag_chain = None

def inicializar_rag():
    global rag_chain
    logger.info("=== [Background] Iniciando procesamiento RAG ===")
    
    # 🚨 LOS IMPORTS PESADOS SE HACEN AQUÍ ADENTRO.
    # Esto evita que el inicio de la app se congele buscando el puerto.
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        documentos_originales = cargar_todos_los_documentos()
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documentos_originales)

        logger.info("Cargando modelo de embeddings de forma aislada...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-albert-small-v2",
            model_kwargs={'device': 'cpu'}
        )

        logger.info("Creando base vectorial FAISS...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",  
            api_key=DEEPSEEK_API_KEY,
            temperature=0.3
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT_TEMPLATE
            | llm
            | StrOutputParser()
        )
        logger.info("=== === [Background] RAG inicializado con éxito! El bot ya puede responder. ===")
        
    except Exception as e:
        logger.error(f"❌ Fallo crítico en hilo de carga RAG: {e}")

def respond(message, history):
    if rag_chain is None:
        return "⏳ El sistema aún se está iniciando en los servidores de Render (Cargando base de conocimiento...). Por favor, intenta de nuevo en 30 segundos."
    try:
        return rag_chain.invoke(message)
    except Exception as e:
        return f"Error: {e}"

def crear_interfaz():
    return gr.ChatInterface(fn=respond, title="Restaurante La Orquídea")

# =====================================================================
# EJECUCIÓN INMEDIATA
# =====================================================================
if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 7860))
    
    # 1. Disparar la carga pesada en paralelo (No bloqueará el puerto)
    hilo = threading.Thread(target=inicializar_rag, daemon=True)
    hilo.start()

    # 2. Arrancar Gradio AL INSTANTE
    demo = crear_interfaz()
    logger.info(f"🚀 Lanzando Gradio INMEDIATAMENTE en el puerto: {puerto}")
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=puerto, 
        share=False
    )
