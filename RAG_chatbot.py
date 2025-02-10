from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import streamlit as st
import os
import tempfile
import whisper
import warnings
import asyncio
import json
from IPython.display import Audio
import edge_tts

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Examining the path of torch.classes raised.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize environment
json_content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
credentials = service_account.Credentials.from_service_account_info(json.loads(json_content))

st.title("📚 DocGenie")
st.markdown("""
    Welcome to the Document Assistant! Upload a PDF document, and you can either type your query or record an audio message. 
    The AI will provide a detailed response based on your document. Please upload only one document 😃
    """)

# File uploader
uploaded_file = st.file_uploader("📄 Upload your PDF file (max 200 MB):", type="pdf")

# Document processing with error handling and caching
if uploaded_file:
    # Sanitize filename
    file_name = os.path.basename(uploaded_file.name)
    
    # Check if we need to reprocess the file
    if ('vectorstore' not in st.session_state or 
        st.session_state.get('processed_file') != file_name):
        
        with st.spinner("🔍 Processing your document..."):
            try:
                # Create temp file
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, file_name)
                
                # Write uploaded file to temp
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and split document
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create vectorstore
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="text-embedding-004", 
                    credentials=credentials
                )
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                # Cache in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_file = file_name
                st.success("✅ Document processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to process document: {str(e)}")
                st.stop()

# Query handling
if 'vectorstore' in st.session_state:
    with st.form("query_form"):
        st.markdown("### 🎤 Record Your Message or Type Your Query")
        audio_data = st.audio_input("Record audio query:", key="audio")
        text_query = st.text_area("Or type your query:")
        submitted = st.form_submit_button("🚀 Submit")

    if submitted:
        try:
            # Process input
            query = text_query
            if audio_data:
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    tmp_audio.write(audio_data.getvalue())
                    tmp_audio_path = tmp_audio.name
                
                # Transcribe audio
                model = whisper.load_model("small")
                result = model.transcribe(tmp_audio_path)
                query = result["text"]
                st.write(f"🎤 You said: {query}")
                os.unlink(tmp_audio_path)  # Cleanup

            # Answer query
            if query:
                # Build QA system
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro-latest", 
                    credentials=credentials
                )
                
                prompt_template = PromptTemplate(
                template="""
                   You are an intelligent AI assistant that prioritizes answering based on the provided documents.  
                    Your responses should be accurate, well-structured, and engaging, ensuring they are grounded in the given content.  
                    
                    If the exact answer is not in the documents, you may:  
                    - Infer a reasonable answer only if there is a strong logical connection to the context.  
                    - Expand on related concepts only if clearly relevant.  
                    - Otherwise, answer using general knowledge** respond with:  
                      "I couldn’t find relevant information in the provided documents. However, based on my general knowledge, here’s what I can suggest."
                      - Highlight technical terms in bold
                      
                    - Use bullet points for steps
                    - If uncertain, state confidence level (80% confident...)

            Now complete the analysis
                    
                    ### Context (from documents):  
                    {context}  
                    
                    ### User Query:  
                    {question}  
                    
                    ### Answer:  
                    """
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(),
                    chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
                )
                
                # Get response
                response = qa_chain.invoke({"query": query})
                
                # Display response
                st.markdown("### 🤖 Response")
                st.write(response["result"])
                
                # Text-to-speech
                if response["result"]:
                    communicate = edge_tts.Communicate(
                        text=response["result"].replace("*", ""),
                        voice="en-US-ChristopherNeural",
                        rate="+10%",
                        pitch="+5Hz"
                    )
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        communicate.save(tmp_file.name)
                        st.audio(tmp_file.name, format="audio/mp3")
                        os.unlink(tmp_file.name)  # Cleanup

        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")

                        
                  
           

    



