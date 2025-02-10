# RAG
Task Description-

Here I created this RAG Chatbot using Langchain, an open-source-frame work (which basically helps us in integration of the AI models into applications)

I referred the [Langchain official documentaion](https://python.langchain.com/docs/introduction/) for using it's functions and had also gone through some [Youtube vedios](https://www.youtube.com/watch?v=1bUy-1hGZpI&t=126s) for getting a clear idea.

So First we need to load data and convert it to an embedding using a vector embedding model.Here i Took the data as a input from the user and used [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/) to load the data and the LLM i used was gemini so i used the Embedding model of the Hugging Faces.

i Strruggeled a lot while using the GOOGLE enbeddings model Later i changed to hugging faces Sentence transformer.
I also spitted the data into chunks using charrecter splitter from Langchain.

In the Later Part of the code i used [FIASS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) method to create vector store of the data loaded and used the similarity search with adding appropraite threshold value to retrieve the relavent documents 
based upon the query.

I also used a function called RetrivalQA which is an inbuilt function in Langchain which helps in reduction of lot of mannual tasks such as Converting query in embedding,searching the appropriate docs and also sending them into ouR LLM model.

Intailly i did not gave any prompt template but later in the expermenting stage found out that giving RAG based prompts Can help us in giving good responses and Also i suggested the model to be a bit creative on topics found relavent in the docs(as per the query) and also Made to use General knowledge if the topic is completely irrelevant(But it will mention it).

Later My main focus was on Adding the voice features in the model so i used Whisper by open-AI for converting speech to text and used microsoft's Edge tts model.I choosed these by doing a bit research about them in the internet.

So I loaded the small model from Whisper since if i was loading the medium or higher my streamlit app was getting timed out and it appeared to be pretty fine.

I also used the tempfile function for storing the file temperarliy while running it.

Later when my base code was ready I tried integrating it Streamlit by taking use of GPT and referring it's documentaion.

I also checked the responses by varying the chunck paramerter and modifying the prompt template 

So here is vedio of my chatbot in text mode and voice mode:

 Context-I uploaded a file of EVS course
 
 Text Mode:
 You can veiw the file called Text-Mode.mp4 in this repository

 Voice Mode-
 You Can veiw the file called Speech-Mode.mp4 in this repository

 For running the code in your local computer-
pip intall all the requirments and packages included in this repository and run the code using streamlit command.

MY deployed APP-[Chatbot](https://l8eaajcwvbluypcdkqrnjl.streamlit.app/)
 

