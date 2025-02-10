from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
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

st.title("📚 Gemini AI Assistant")
st.markdown("""
    Welcome to the Gemini AI Assistant! Upload a PDF document, and you can either type your query or record an audio message. 
    The AI will process your input and provide a detailed response only based upon on your document and upload only one document 😃
    """)

uploaded_file = st.file_uploader("📄 Upload your PDF file (max 200 MB):", type="pdf")

if uploaded_file:
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    documents = PyPDFLoader(temp_file_path).load()
    text_1 = "\n".join([doc.page_content for doc in documents])
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text_1)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", credentials=credentials,temperature=0.3)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)
    vectorstore = FAISS.from_documents(documents, embeddings)

    with st.form("my_form"):
        st.markdown("### 🎤 Record Your Message or Type Your Query(Do only one)")
        audio_record = st.audio_input("🎙️ Record your message:")
        text_query = st.text_area("✍️ Or type your query here:")
        submitted = st.form_submit_button("🚀 Submit")

    if submitted:
        try:
            if audio_record:
                # Handle audio input
                temp_audio_path = os.path.join(temp_dir, "recorded_audio.wav")
                with open(temp_audio_path, "wb") as temp_audio_file:
                    temp_audio_file.write(audio_record.read())
            
                with st.spinner("Processing..."):
                    transcription = whisper.load_model("small").transcribe(temp_audio_path, language="en")
                    query = transcription["text"]
                st.write("🎤 You said:", query)
            else:
                
                query = text_query
            prompt_template= PromptTemplate(
                template="""
               You are an intelligent AI assistant that *prioritizes answering based on the provided documents*.  
                Your responses should be *accurate, well-structured, and engaging*, ensuring they are grounded in the given content.  
                
                If the exact answer is *not in the documents*, you may:  
                - *Infer* a reasonable answer only if there is a strong logical connection to the context.  
                - *Expand* on related concepts *only if clearly relevant*.  
                - Otherwise, answer using general knowledge** respond with:  
                  "I couldn’t find relevant information in the provided documents. However, based on my general knowledge, here’s what I can suggest."  
                
                ### *Context (from documents):*  
                {context}  
                
                ### *User Query:*  
                {question}  
                
                ### *Answer:*  
                """
            )
  
            retriever = vectorstore.as_retriever(search_kwargs={"k": 8},similarity_score_threshold=0.7)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever,chain_type_kwargs={"prompt": prompt_template})
            response = qa_chain.invoke(query)

            # Display response
            st.markdown("### 🤖 Generated Response:")
            st.write(response["result"])

            # Text-to-speech conversion
            if "result" in response:
                text = response["result"]
                text = text.replace("**", " ").replace("*", " ").replace("_", " ")
                
                VOICE = "en-US-ChristopherNeural"
                RATE = "+10%"
                PITCH = "+5Hz"

                async def generate_speech():
                    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
                    await communicate.save("human_like_audio.mp3")

                try:
                    asyncio.run(generate_speech())
                    if os.path.exists("human_like_audio.mp3"):
                        st.audio("human_like_audio.mp3", autoplay=False)
                except Exception as e:
                    st.error(f"Error generating speech: {e}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
                
                  
           

    



