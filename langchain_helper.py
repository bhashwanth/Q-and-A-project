from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import GooglePalm
llm = GooglePalm(google_api_key = os.environ["GOOGLE_API_KEY"] , temperature = 0)

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path = "faiss_index"
def create_vector_db():
    loader = CSVLoader(file_path='/Users/bhashreddy/Downloads/codebasics_faqs.csv', source_column='prompt',
                       encoding='latin-1')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data , embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path , instructor_embeddings)

    retriever = vectordb.as_retriever(score_threshold = 0.7)

    prompt_template = """ Given the following context and question, generate a response based on this context.If the answer is not found in the context say 'I dont Know' in the respone.Please Dont assume things.
    CONTEXT : {context}
    QUESTION : {question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key='query',
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

if __name__ == "__main__":
    chain = get_qa_chain()

    print(chain("Do you have internship opportunities and will you offer EMI payments?"))