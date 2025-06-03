import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load text embedding model
model_name = "BAAI/bge-large-en-v1.5"    
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)



# reload vector database
store_dir = 'BAAI_db/chroma_vector_store'

store = Chroma(
        persist_directory=store_dir,
        embedding_function=embeddings_model
    )



# define working loop and LLM
retriever = store.as_retriever(search_kwargs={"k": 3})

#TEMPLATE !!!!
prompt_template = '''Here is a question about Professor Whinston's research papers. To use the following context from original paper to answer my question

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


queries = ['I am wondering if Professor Whinston had research on politeness bias',
           'In the article "Is Best Answer Really the Best Answer? The Politeness Bias", how does the the researchers measure if an answer is polite or not?',
           'In the article "Is Best Answer Really the Best Answer? The Politeness Bias", how was the measurement of impolityness obtained?'
           'why do we care about the second-person proun "you" when measuring the politeness'
           'How does the author precisely construct the indicator "impoliteness" for an answer?'
           "Why do we care about answer provider's reputation when studying politeness bias?  "
    
]

qa_pair = []
for query in queries:
    respond = qa_chain.invoke({"query":query})
    result = {
            "query": query,
            "result": respond["result"]
            
        }
    qa_pair.append(result)
    print(respond["result"])

print(qa_pair)

