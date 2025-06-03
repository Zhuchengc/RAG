from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### load pdf files ###
documents = []

pdf_folder_path = "papers"
failed_pdf = []
success_count = 0
for root, dirs, files in os.walk(pdf_folder_path):
    for file in files:
        try:

            pdf_path = os.path.join(root, file)

            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            success_count += 1

        except Exception as e:
            print(f"error {file}:{e}")
            failed_pdf.append({"file": file, "error": str(e)})


print(len(documents))
print(failed_pdf)
print(success_count)
#check content
#for i, doc in enumerate(documents[50:80]):
    #print(doc)


## text chunk  ####

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} text chunks.")

#print(split_docs[5].page_content)
#print(split_docs[5].metadata)



### text embedding


model_name = "BAAI/bge-large-en-v1.5"
#"sentence-transformers/msmarco-MiniLM-L-6-v3" 
#"BAAI/bge-large-en-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


#embeddings_model.embed_documents(chunks)

texts = []
for i, doc in enumerate(chunks):
    print(doc.page_content)
    texts.append(doc.page_content)

text_embeddings = embeddings_model.embed_documents(texts)

print(len(text_embeddings[0]))



### text vectors storage ###

store_dir = 'BAAI_db/chroma_vector_store'

store = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings_model,
    persist_directory = store_dir
)
store.persist()





### defining working loop ###
retriever = store.as_retriever(search_kwargs={"k": 3})

#TEMPLATE !!!!
prompt_template = '''To use the following context to answer my question

context:{context}
Question:{question}
'''

prompt = PromptTemplate(input_variables=["context","question"], template=prompt_template)

#define LLM
llm = Ollama(model="llama3")



#define QA chian
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# change query here!!
query = "what is the advantage of a network-embedded prediction market?"

result = qa_chain.invoke({"query":query})
print(result["result"])



### Reload database for query !!!!!


model_name = "sentence-transformers/msmarco-MiniLM-L-6-v3" 
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

store_dir = 'db/chroma_vector_store'
vectorstore = Chroma(
        persist_directory=store_dir,
        embedding_function=embeddings_model
    )

print(vectorstore._collection.count())


retriever = store.as_retriever(search_kwargs={"k": 3})


prompt_template = '''To use the following context to answer my question

context:{context}
Question:{question}
'''

prompt = PromptTemplate(input_variables=["context","question"], template=prompt_template)

#define LLM
llm = Ollama(model="llama3")



#define QA chian
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# change query here!!
query = "what is the advantage of a network-embedded prediction market?"

result = qa_chain.invoke({"query":query})
print(result["result"])