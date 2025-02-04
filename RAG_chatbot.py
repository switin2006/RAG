from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Set credentials
json_content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
credentials = service_account.Credentials.from_service_account_info(json.loads(json_content))

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTextArea>textarea {
        border-radius: 5px;
        padding: 10px;
    }
    .stAudio {
        border-radius: 5px;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📚 Gemini AI Assistant")
st.markdown("""
    Welcome to the Gemini AI Assistant! Upload a PDF document, and you can either type your query or record an audio message. 
    The AI will process your input and provide a detailed response.
    """)

# File uploader
uploaded_file = st.file_uploader("📄 Upload your PDF file (max 200 MB):", type="pdf")

if uploaded_file:
    st.write("📂 File Name:", uploaded_file.name)
    
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Load the PDF file
    documents = PyPDFLoader(temp_file_path).load()
    text_1 = "\n".join([doc.page_content for doc in documents])
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=50  
    )
    chunks = text_splitter.split_text(text_1)
    
    # Initialize Gemini AI
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", credentials=credentials)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # User input form
    with st.form("query_form"):
        st.markdown("### 🎤 Record Your Message or Type Your Query")
        audio_record = st.audio_input("🎙️ Record your message:")
        text_query = st.text_area("✍️ Or type your query here:")
        submitted = st.form_submit_button("🚀 Submit")
        
    if submitted:
        if audio_record:
            # Save the recorded audio to a temporary location
            temp_audio_path = os.path.join(temp_dir, "recorded_audio.wav")
            with open(temp_audio_path, "wb") as temp_audio_file:
                temp_audio_file.write(audio_record.read())
            
            # Transcribe the audio
            transcription = whisper.load_model("small").transcribe(temp_audio_path, language="en")
            query = transcription["text"]
            st.write("🎤 You said:", query)
        else:
            query = text_query
        
        # Retrieve and generate response
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        
        try:
            response = qa_chain.invoke(query)
            st.markdown("### 🤖 Generated Response:")
            st.write(response["result"])
            
            # Convert response to speech
            text = response["result"]
            text = text.replace("**", " ").replace("*", " ").replace("_", " ")
            
            # Customize voice and speech parameters
            VOICE = "en-US-ChristopherNeural"  # Deep male voice
            RATE = "+10%"  # Speed adjustment
            PITCH = "+5Hz"  # Slightly lower pitch for warmth

            # Generate speech with adjustments
            async def generate_speech():
                communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
                await communicate.save("human_like_audio.mp3")

            asyncio.run(generate_speech())

            # Play the generated audio
            if os.path.exists("human_like_audio.mp3"):
                st.audio("human_like_audio.mp3", autoplay=True)
            else:
                st.write("Audio file not found.")
                
        except Exception as e:
            st.write(f"❌ An error occurred: {e}")
            
                
                  
           

    



