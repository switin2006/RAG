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
# used to supress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Examining the path of torch.classes raised.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setting up the environment
json_content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
credentials = service_account.Credentials.from_service_account_info(json.loads(json_content))

st.title("📚 DocGenie")
st.markdown("""
    Welcome to the Document Assistant! Upload a PDF document, and you can either type your query or record an audio message. 
    The AI will process your input and provide a detailed response only based upon on your document and upload only one document 😃
    """)
#Creating a file Uploader
uploaded_file = st.file_uploader("📄 Upload your PDF file (max 200 MB):", type="pdf")
#Creating temp file to get the path of the Uploaded document since we don't know where is it stored during the run.
if uploaded_file:
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
#Extracting the text from the documents 
    documents = PyPDFLoader(temp_file_path).load()
    text_1 = "\n".join([doc.page_content for doc in documents])
    #Creating chunks and converting tthem into doc object (to be passable to FIASS)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text_1)
    doc_chunks = [Document(page_content=chunk) for chunk in chunks]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", credentials=credentials)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", credentials=credentials)
    #storing the documents as embeddings in FIASS
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
#creating a form for taking inputs from the user
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
                
                transcription = whisper.load_model("small").transcribe(temp_audio_path, language="en")
                query = transcription["text"]
                st.write("🎤 You said:", query)
            else:
                
                query = text_query
            #Using prompt Template
            prompt_template= PromptTemplate(
                    template="""
                   You are an intelligent AI assistant that prioritizes answering based on the provided documents.  
                    Your responses should be accurate, well-structured, and engaging, ensuring they are grounded in the given content.  
                    
                    If the exact answer is not in the documents, you may:  
                    - Infer a reasonable answer only if there is a strong logical connection to the context.  
                    - Expand on related concepts only if clearly relevant.  
                    - Otherwise, answer using general knowledge** respond with:  
                      "I couldn’t find relevant information in the provided documents. However, based on my general knowledge, here’s what I can suggest."
                      - Highlight technical terms in *bold*
                      
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

           
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
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

                        
                  
           

    



