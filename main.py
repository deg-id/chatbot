import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

with open("cafeteria.txt", "r", encoding="utf-8") as f:
    documento = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(documento)

vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Retorna los 4 fragmentos mas relevantes
)

prompt = ChatPromptTemplate.from_template("""Eres el asistente virtual de Café Aurora. Tu trabajo es responder preguntas
de los clientes ÚNICAMENTE usando la información proporcionada en el contexto.

Reglas estrictas:
1. SOLO responde con información que esté en el contexto.
2. Si la pregunta no se puede responder con el contexto, di:
   "Lo siento, no tengo esa información. Te recomiendo contactarnos
   por WhatsApp al +56 9 8765 4321 o por email a contacto@cafeaurora.cl"
3. Sé amable, conciso y útil.
4. Si preguntan precios, siempre menciona el precio exacto.
5. Responde en español.

Contexto:
{context}

Pregunta del cliente: {question}

Respuesta:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def respond(message, history):
    response = rag_chain.invoke(message)
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="Café Aurora - Asistente Virtual",
    description="Pregúntame sobre nuestro menú, horarios, ubicación, eventos y más.",
    examples=[
        "¿Cuál es el horario de atención los sábados?",
        "¿Tienen opciones veganas?",
        "¿Cuánto cuesta un cappuccino?",
        "¿Hacen delivery?",
        "¿Tienen wifi?",
    ],
)

if __name__ == "__main__":
    print(f"Documento cargado: {len(chunks)} fragmentos indexados")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme="soft"
    )
