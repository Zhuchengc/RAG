# RAG
This is a RAG system based on Professor Whinston's research papers

Download original papers from https://www.kaggle.com/datasets/dwdddwij/professor-whinstons-research-papers/data

See libraries in install list.txt

When you have all the .py script and 'papers' folder in same working directory, run langchain_rag.py while having Ollama running. This should create a folder called 'BAAI_db' and finally print a answer in terminal. This script may take some time to run, depending on the GPU performance(it took me 2 hours).

After that, execute rag_backend.py and paste 'streamlit run streamlit_demo.py' in a new terminal in your IDE. this should create you a website and you can chat with the RAG assitant now.
