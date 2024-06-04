from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.llms import Bedrock
from langchain_community.retrievers import AmazonKendraRetriever
import os

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

# Configure streamlit app
st.set_page_config(page_title = "UNO Bot")
st.title("UNO Bot")

# Define the retriever
# retriever = AmazonKnowledgeBasesRetriever(
#     knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID"),
#     retrieval_config = {"vectorSearchConfiguration": {"numberOfResults": 4}},
# )

retriever = AmazonKendraRetriever(
  index_id= os.getenv("KENDRA_INDEX_ID"),
  top_k= 5
)

# Define model parameters
model_kwargs_claude = {
  'temperature' : 0,
  'top_k' : 10,
  'max_tokens_to_sample' : 700
}

# Configure llm
llm = Bedrock(model_id = "anthropic.claude-instant-v1", model_kwargs = model_kwargs_claude)

# Set up message history
mesgs = StreamlitChatMessageHistory(key ='langchain_messages')
memory = ConversationBufferMemory(chat_memory = mesgs, memory_key = 'history', output_key = 'answer', return_messages = True)
if len(mesgs.messages) == 0:
  mesgs.add_ai_message('Do you have any questions about UNO?')


#Creating the template   
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from a knowledge base. 
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 

{context}

{chat_history}

Question: {question}

Assistant:
"""

#Configure prompt template
prompt_template = PromptTemplate(
  input_variables= ['context', 'chat_history', 'question'],
  template = my_template
)


#Configure the chain
qa = ConversationalRetrievalChain.from_llm(
  llm = llm,
  retriever = retriever,
  return_source_documents = True,
  combine_docs_chain_kwargs = {"prompt" : prompt_template},
  memory = memory,
  condense_question_prompt = CONDENSE_QUESTION_PROMPT
)

#Render current messages from StreamlitChatMessageHistory
for mesg in mesgs.messages:
  st.chat_message(mesg.type).write(mesg.content)


#If user inputs a new prompt, generate and draw a new response
if prompt:= st.chat_input():
  st.chat_message('human').write(prompt)

  # # invoke the llm - messages are saved to history automatically by langchain
  # output = qa.invoke({'query':prompt})

  # # adding messages to memory
  # memory.chat_memory.add_user_message(prompt)
  # memory.chat_memory.add_ai_message(output['result'])

  # # display output
  # st.chat_message("ai").write(output['result'])

  # Invoke the model
  output = qa.invoke({'question' : prompt, 'chat_history' : memory.load_memory_variables({})})
    
  # Display the output
  st.chat_message("ai").write(output['answer'])  