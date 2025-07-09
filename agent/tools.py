from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from constants import logger, PROMPT
from .config import get_llm

#root- const, aget-chain,aget

@tool
def answer_using_rag(query: str) -> dict:
    """
    Use the knowledge base (via RAG) to answer the user's query.

    Falls back to a general response if the RAG result is insufficient.
    """
    try:
        if not os.path.exists(VECTORSTORE_DIR):
            logger.info("Vector store not found. Triggering update.")
            update_vector_store()

        vector_store = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings=get_embadings(),
            allow_dangerous_deserialization=True
        )

        results = vector_store.similarity_search(query=query, k=3)
        context = " ".join([doc.page_content for doc in results]) if results else ""
        logger.info(f"Retrieved {len(results)} relevant documents for query: '{query}'")

        if not context:
            logger.warning("No relevant context found. Falling back to general query.")
            return general_query.invoke({"query": query})

        chain = PROMPT | get_llm() | StrOutputParser()
        result = chain.invoke({"context": context, "question": query})

        if "does not provide information" in result.lower() or len(result.strip()) < 10:
            logger.warning("RAG output is vague. Falling back to general query.")
            return general_query.invoke({"query": query})

        return {"input": query, "output": result}

    except Exception as e:
        logger.exception("RAG-based answering failed.")
        return {"input": query, "output": f"An error occurred: {str(e)}"}

@tool
def answer_billing_query(query: str) -> str:
    """
    Answer billing related query.

    """
    promt = "you are an helpfull asistant to give billing answers to the customers"
    promt1 = ChatPromptTemplate.from_messages(
       [ ("system",promt)]
    )

    llm = get_llm()
    chain = promt1 | llm | StrOutputParser()
    result = chain.invoke({"question": query})
    return {"input": query, "output": f"{result}"}

@tool
def answer_technical_query(query: str) -> str:
    """Answer technical support related query."""
    return {"input": query, "output": f"This is a technical support response to: {query}"}

@tool
def general_query(query: str) -> str:
    """
    Generates a generic response when no specific tool applies.
    """
    parser = StrOutputParser()
    chain = get_llm() | parser
    output = chain.invoke(query)
    return {"input": query, "output": f"This is a general response to: '{output}'"}


@tool
def escalate_to_human(query: str) -> str:
    """ 
    Escalating to live agent
    """
    return {"input": query, "output": f"Escalating to human support for query: '{query}'"}

