import tempfile
import streamlit as st

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredMarkdownLoader


MODEL_NAME = "llama3"



# SEGMENT 1 : INDEXATIONS (PDF & md)


def build_vectorstore_from_files(uploaded_files):
    """Prend une liste de fichiers (PDF/MD), extrait et fusionne tous leurs morceaux."""
    if not uploaded_files:
        raise ValueError("No files provided.")
    
    tous_les_chunks = []
    

    for uploaded_file in uploaded_files:
        nom_fichier = uploaded_file.name
        extension = nom_fichier.split(".")[-1].lower()
        
        if extension not in ["pdf", "md"]:
            st.warning(f"⚠️ Format ignoré pour le fichier {nom_fichier} (uniquement PDF et MD).")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if extension == "pdf":
            loader = PyPDFLoader(tmp_path)    
        elif extension == "md":
            loader = UnstructuredMarkdownLoader(tmp_path)
            
        docs = loader.load()
        if not docs:
            continue
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)
        
        tous_les_chunks.extend(chunks)

    if not tous_les_chunks:
        raise ValueError("No chunks could be extracted from any of the uploaded files.")
    
    vectordb = Chroma.from_documents(
        documents=tous_les_chunks,
        embedding=get_embeddings(),
    )
    return vectordb


# SEGMENT 2 : EMBEDDING & CONFIGURATION DES MODÈLES


@st.cache_resource
def get_llm():
    """Initialise le grand modèle de langage (Llama 3) avec une basse température."""
    return ChatOllama(model=MODEL_NAME, temperature=0.2)

@st.cache_resource
def get_embeddings():
    """Initialise le modèle d'embedding (nomic) pour la vectorisation de texte."""
    return OllamaEmbeddings(model="nomic-embed-text")



# SEGMENT 3 : PIPELINE RAG 


def make_rag_chain(vectordb, k: int, answer_style: str):
    # Le composant Retriever va chercher les 'k' morceaux de texte les plus pertinents
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    def join_context(question: str) -> str:
        docs = retriever.invoke(question)
        if not docs:
            return "No relevant context found in the document."
        return "\n\n-----\n\n".join(d.page_content for d in docs)
    
    # Modèle de consigne stricte imposé au LLM
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions based ONLY on the given PDF.
If the answer is not clearly in the PDF, say:
"I don't see this clearly in the document" 
User prefers: {answer_style} answers

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = get_llm()

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": join_context,
            "answer_style": lambda _: answer_style,
        }
        | prompt
        | llm
    )
    return rag_chain, retriever



# INTERFACE APPLICATION


def init_sessions():
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None  
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="RAG: Chat With your files",
        page_icon="📄",
        layout="wide",
    )

    init_sessions()  
    st.title("📄 Chat with your files")
    st.caption(
        "Upload a PDF, we'll build a local vector store, and you can ask questions "
        "grounded in that document. Runs fully with Ollama."
    )

    with st.sidebar:
        st.header("⚙️ RAG Settings")
        top_k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=8, value=4)
        answer_style = st.selectbox(
            "Answer style",
            ["short and crisp", "Detailed"],
            index=0,
        )

        if st.button("Clear chat and reset"):
            st.session_state.chat_history = []
            st.session_state.vectordb = None
            st.success("State cleared. Upload a file again.")

    col_left, col_right = st.columns([1, 2])  

    with col_left:
        st.subheader("Upload and Process file")
        uploaded_file = st.file_uploader(
            "Upload a PDF/md file",
            type=["pdf", "md"],
            accept_multiple_files=True,
            help="select all files needed"
        )

        if st.button("⚙️ Process Documents"):
            if not uploaded_file: 
                st.warning("Please upload at least one file first.")
            else:
                try:
                    with st.spinner("Reading, chunking, and embedding all your files..."):
                        vectordb = build_vectorstore_from_files(uploaded_file)
                        st.session_state.vectordb = vectordb
                    st.success(f"🎉 {len(uploaded_file)} files processed together! You can ask questions.")
                except Exception as e:
                    st.error(f"ERROR while processing files: {e}")

        if st.session_state.vectordb is None:
            st.info("Upload documents and click **process documents** to get started")
        
        st.markdown("---")
        st.subheader("Example questions")
        st.markdown(
            "- *What is the main conclusion of this document?*\n"
            "- *List the key recommendations.*"
        )

    with col_right:
        st.subheader("Ask Questions About Your PDF")
        if st.session_state.vectordb is None:
            st.warning("No file processed yet. Upload and process a file on the left.") 
            return
            
        question = st.text_input(
            "Type your question",
            placeholder="Example: What are the key takeaways of this document?"
        )

        ask_clicked = st.button("• Ask")

        if ask_clicked and question.strip():
            try:
                rag_chain, retriever = make_rag_chain(
                    st.session_state.vectordb,
                    k=top_k,
                    answer_style=answer_style,
                )
                with st.spinner("Thinking with RAG..."):
                    res = rag_chain.invoke(question)
                    answer = getattr(res, "content", str(res))

                    docs = retriever.invoke(question)
                    sources = []
                    for d in docs:
                        sources.append(
                            {
                                "page": d.metadata.get("page", "Unknown"),
                                "snippet": d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""),
                            }
                        )
                    st.session_state.chat_history.append(
                        {"q": question, "a": answer, "sources": sources}
                    )
            except Exception as e:
                st.error(f"Error while generating answer: {e}")  

        # Affichage du Chat
        for item in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(item["q"])  
            with st.chat_message("assistant"):
                st.markdown(item["a"])  
                if item["sources"]: 
                    with st.expander("View sources from PDF"):
                        for i, s in enumerate(item["sources"], start=1):
                            st.markdown(
                                f"**Source {i}** - Page `{s['page']}`\n\n"
                                f"{s['snippet']}\n"
                            )

if __name__ == "__main__":
    main()