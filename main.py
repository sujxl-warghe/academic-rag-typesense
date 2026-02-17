import os
import typesense
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Typesense
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# Load env
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.2
)

# --------------------------------------------------
# Embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Typesense Client
# --------------------------------------------------
typesense_client = typesense.Client({
    "nodes": [{
        "host": os.getenv("TYPESENSE_HOST"),
        "port": os.getenv("TYPESENSE_PORT"),
        "protocol": os.getenv("TYPESENSE_PROTOCOL", "http")
    }],
    "api_key": os.getenv("TYPESENSE_API_KEY"),
    "connection_timeout_seconds": 2
})

# --------------------------------------------------
# Vector Store (WORKING VERSION)
# --------------------------------------------------
vectorstore = Typesense(
    typesense_client=typesense_client,
    name="MDM_RAG",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --------------------------------------------------
# Prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are an academic assistant.
Answer strictly using the given context.
Do NOT use outside knowledge.
Write in exam-oriented format.

Context:
{context}

Question:
{question}

Answer:
"""
)

# --------------------------------------------------
# RAG Chain
# --------------------------------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("\n📚 Academic RAG Question Answering System")
    print("Type 'exit' to quit\n")

    while True:
        query = input("❓ Enter your question: ")

        if query.lower() == "exit":
            break

        answer = rag_chain.invoke(query)
        print("\n✅ Answer:\n", answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
