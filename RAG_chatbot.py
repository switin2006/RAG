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
from IPython.display import Audio
import edge_tts

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Examining the path of torch.classes raised.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set Groq API key
os.environ["GROQ_API_KEY"] = "your-groq-api-key"

st.title("📚 DocGenie")
st.markdown("""
    Welcome to the Document Assistant! Upload a PDF document, and you can either type your query or record an audio message. 
    The AI will process your input and provide a detailed response based only upon your document. Please upload only one document. 😃
    """)

# File uploader
uploaded_file = st.file_uploader("📄 Upload your PDF file (max 200 MB):", type="pdf")

if uploaded_file:
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Extract text from the document
    documents = PyPDFLoader(temp_file_path).load()
    text_1 = "\n".join([doc.page_content for doc in documents])

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text_1)
    doc_chunks = [Document(page_content=chunk) for chunk in chunks]

    # Initialize Groq LLM
    from langchain_groq import ChatGroq  # Assuming Groq has a LangChain integration
    llm = ChatGroq(model="groq-model-name")  # Replace with the actual Groq model name

    # Initialize embeddings (you can still use Google's embeddings or another provider)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Store the documents as embeddings in FAISS
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)

    # Form for user input
    with st.form("my_form"):
        st.markdown("### 🎤 Record Your Message or Type Your Query (Do only one)")
        audio_record = st.file_uploader("🎙️ Upload your audio message:", type=["wav", "mp3"])
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

            # Prompt template
            prompt_template = PromptTemplate(
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

                Now complete the analysis
                
                ### Context (from documents):  
                {context}  
                
                ### User Query:  
                {question}  
                
                ### Answer:  
                """
            )

            # Retrieve and generate response
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt_template})
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

                        
                  
           

    



