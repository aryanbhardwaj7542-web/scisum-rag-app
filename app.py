import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. UI Configuration and Setup
st.set_page_config(page_title="SciSum RAG Assistant", layout="wide")
st.title("🔬 SciSum RAG: Scientific Paper Assistant")
st.write("Upload a research paper, parse its contents, and chat with it securely using custom RAG workflows.")

# Sidebar for secure API key configuration
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    st.info("Obtain a free developer key from Google AI Studio.")

# Execute pipeline only if the API key is provided
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # 2. File Upload Interface
    uploaded_file = st.file_uploader("Upload your research paper (PDF format only)", type="file")

    if uploaded_file is not None:
        # Save file to a temporary local path so the PyPDFLoader can access it
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document... Extracting text and constructing vector shards."):
            try:
                # 3. Document Parsing Step
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()

                # 4. Text Chunking Step
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # 5. Embedding Generation and FAISS Vector Storage Setup
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
                
                st.success("Document vectorized successfully! Pipeline is active.")
            except Exception as e:
                st.error(f"Error initializing RAG components: {e}")
            finally:
                # Clean up local temporary file safely
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        # 6. LLM Brain Setup
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        
        # System instructions to minimize AI hallucinations
        system_prompt = (
            "You are an elite academic research assistant. Answer the user's inquiry "
            "using exclusively the context block provided below. If the solution cannot be "
            "explicitly deduced from the text, reply with: 'I am sorry, but that specific detail "
            "is not contained within the provided paper.' Do not extrapolate or create external data.\n\n"
            "Context Block:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 7. LCEL RAG Chain (Highly stable, no legacy import required)
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 8. Interactive Conversation Layer
        st.divider()
        st.subheader("💬 Ask Your Research Assistant")
        user_query = st.text_input("Pose a question regarding methodologies, findings, or metrics:")

        if user_query:
            with st.spinner("Analyzing semantic vectors..."):
                response = rag_chain.invoke(user_query)
                st.markdown("### 🤖 Engine Output:")
                st.write(response)
else:
    st.warning("Please input your Gemini API Key in the left sidebar to unlock the application framework.")
