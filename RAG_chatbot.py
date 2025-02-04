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
import os
import json
#Used to reduce the runtime error
warnings.filterwarnings("ignore", category=UserWarning, message="Examining the path of torch.classes raised.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#I am setting credentials
json_content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
credentials = service_account.Credentials.from_service_account_info(json.loads(json_content))
uploaded_file = st.file_uploader("Upload your PDF file:", type="pdf")
#Asking the user the upload the pdf file of 200 MB limit
if uploaded_file:
    st.write("File Name:", uploaded_file.name)
    
#Here we while write the code do not know thw file path so we cannot process it further but we can know the file path by storing it in temparary variable
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

   #For retrieving path
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    #Loding the pdf file using the temp path
    documents = PyPDFLoader(temp_file_path).load()
    #Retrieving text from it 
    text_1 = "\n".join([doc.page_content for doc in documents])
    #Processing the retrieved text in chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=50  
    )
    chunks = text_splitter.split_text(text_1)
    # i am using gemini ai
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", credentials=credentials)
    #Creating vector embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)
    #Creating the vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    st.title("Assistant with Gemini AI ")
    with st.form("my_form"):
          Audio_record=st.audio_input("Record Your Message")
          text_2=st.text_area("Enter your query")
          submitted=st.form_submit_button("Submit")
          
        
    if submitted:
            if not Audio_record:
                query = text_2
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
                
                try:
                    response = qa_chain.invoke(query)
                    st.write("\nGenerated Response:\n", response)
                except Exception as e:
                            
                            st.write(f"An error occurred: {e}")
            if  Audio_record:
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir,"recorded_audio.wav")

                # Write the uploaded file to the temporary location
                with open(temp_file_path, "wb") as temp_file:
                      
                      temp_file.write(Audio_record.read())
                
                transcription =whisper.load_model("small").transcribe(temp_file_path,language="en")
                query=transcription["text"]
                st.write(query)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
                
                try:
                    response = qa_chain.invoke(query)
                    st.write("\nGenerated Response:\n", response)
                except Exception as e:
                            
                            st.write(f"An error occurred: {e}")
                from IPython.display import Audio
                import edge_tts

                # Your text
                text = response["result"]
                text = text.replace("**", " ")
                text = text.replace("*", " ")
                text = text.replace("_", " ")
                # Customize voice and speech parameters
                VOICE = "en-US-ChristopherNeural"  # Deep male voice
                RATE = "+10%"  # Speed adjustment (-20% to +20% for natural pacing)
                PITCH = "+5Hz"  # Slightly lower pitch for warmth

                # Generate speech with adjustments
                async def generate_speech():
                    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
                    await communicate.save("human_like_audio.mp3")

                # Run the async function
                asyncio.run(generate_speech())

                # Ensure the file is fully written before attempting to play it
                if os.path.exists("human_like_audio.mp3"):
                    st.audio("human_like_audio.mp3", autoplay=True)
                else:
                    st.write("Audio file not found.")
            
                
                  
           

    



