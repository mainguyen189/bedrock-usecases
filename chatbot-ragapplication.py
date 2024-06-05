import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Configure streamlit app
st.set_page_config(page_title="Catan Bot", page_icon="📖")
st.title("📖 Catan Bot")

# Define vectorstore
global vectorstore_faiss

# Define convenience functions
# cache to prevent resource rerun when app reload
@st.cache_resource
def config_llm():
    client = boto3.client('bedrock-runtime')

    model_kwargs = { 
        "max_tokens_to_sample": 512,
        "temperature":0.1,  
        "top_p":1
    }  

    model_id = "anthropic.claude-instant-v1"
    llm = Bedrock(model_id=model_id, client=client)
    llm.model_kwargs = model_kwargs
    return llm

@st.cache_resource
def config_vector_db(filename):
    client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(client=client)
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

# def vector_search(query):
#     docs = vectorstore_faiss.similarity_search_with_score(query)
#     info = ""
#     for doc in docs:
#         info+= doc[0].page_content+'\n'
#     return info    


# Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db('https://www.catan.com/sites/default/files/2021-06/catan_c_k_2020_rule_book_200708.pdf')

# Set up memory
mesgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(mesgs.messages) == 0 :
    mesgs.add_ai_message("How can I help you about Catan?")

# Creating the template   
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from Catan player. 
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Only answer the question. Do not say things like "according to the instruction handbook or according to the information provided...".
    
    <Information>
    {info}
    </Information>
    

    {input}

Assistant:
"""

# Configure prompt template
prompt_template = PromptTemplate(
    input_variables=['input', 'info'],
    template=my_template
)

#Create llm chain
question_chain = LLMChain(
    llm=llm,
    prompt= prompt_template,
    output_key='answer'
)

# # Get question, peform similarity search, invoke model and return result
# while True:
#     question = input("\nAsk a question about the Catan board game")

#     # perform similarity search
#     info = vector_search(question)

#     # invoke model with context
#     output = question_chain.invoke({
#         'input' : question,
#         'info' : info
#     })

#     print(output['answer'])

# Render current messages from StreamlitChatMessageHistory
for mesg in mesgs.messages:
    st.chat_message(mesg.type).write(mesg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # retrieve relevant documents using a similarity search
    docs = vectorstore_faiss.similarity_search_with_score(prompt)
    info = ""
    for doc in docs:
        info += doc[0].page_content + '\n'

    # invoke llm
    output = question_chain.invoke({"input" : prompt, "info" : info})

    # adding messages to history
    mesgs.add_user_message(prompt)
    mesgs.add_ai_message(output['answer'])

    # display the output
    st.chat_message("ai").write(output['answer'])
