# Code to be run only once inorder to create vector stores of our docs
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from main import get_secret
load_dotenv()
#Load the document
doc2txt=Docx2txtLoader(file_path='files/alex_characteristics.docx')
pages=doc2txt.load()


#Markdown Splitters
markdownsplit = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Character"),             # For the main title "Alex: Social Experiment Participant Identity and Conversation Guide"
        ("##", "Section"),             # For "Identity Characteristics", "Conversation Flow", "Post Core Message Response", "Conclusion", "Transparency"
        ("###", "Sub-section"),        # For "Greeting", "Quick Warm-up", "Transition", "Core Message", "If User Says \"Yes\"", etc.
    ],
    strip_headers=False
)

splitted_pages=markdownsplit.split_text(pages[0].page_content)

#Recursive Splitters
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
page_split = recursive_splitter.split_documents(splitted_pages)
page_split

#Embeddings
api_key=get_secret("OPENAI_API_KEY")
embeddings=OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)


#Create the vecor stores
vector_store= Chroma.from_documents(
        documents=page_split,
        embedding=embeddings,
        #the persist directory is the name of the vector file
        persist_directory="./alex_characteristics",
        collection_name="social_experiment",
        
    )
