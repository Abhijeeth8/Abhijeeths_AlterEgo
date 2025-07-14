# from importlib import metadata
# from xml.dom.minidom import Document
from dotenv import load_dotenv
from openai import api_key
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document 
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore, Pinecone
from langchain_core.output_parsers import MarkdownListOutputParser
# import sendgrid
import os
# from sendgrid.helpers.mail import Content, Email, To, Mail, Subject
from langchain.tools import tool

from gmail_drafts import create_draft, create_message, gmail_authenticate
import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

def authenticate_gmail():
  service = gmail_authenticate()
  return service

def draft_creation(service, user_id, message_body):
  draft = create_draft(service, user_id, message_body)
  return draft

def mail_creation(sender, to, subject, message_text):
  mail_details = create_message(sender, to, subject, message_text)
  return mail_details

@tool
def compose_and_draft_mail(sender:str, reciepient:str, subject:str, message_text:str) -> str:
  """
  This tool drafts emails in gmail with the provided subject, body and with the correct sender and reciepient information
  """
  print("Message is being drafted.....")
  service = authenticate_gmail()
  mail_content = mail_creation(sender, reciepient, subject, message_text)
  draft_mail = draft_creation(service, "me", mail_content)

  return "Mail has been drafted successfully in Abhijeeth's gmail."





# @tool
# def send_email(subject:str, body: str):
#     """ If the interviewer wants to connect this tool can send an email with the given subject and body"""

#     sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
#     from_email = Email("abhijeethkollarapu@gmail.com") 
#     to_email = To("abhijeeth1200@gmail.com") 
#     sub = Subject(subject=subject)
#     content = Content("text/plain", body)
#     mail = Mail(from_email, to_email, sub, content).get()
#     response = sg.client.mail.send.post(request_body=mail)
#     return {"status": "success", "message":f"Mail has beed successfully sent."}


# uk_questions = []
# @tool
# def unknown_questions_list(question:str) -> str:
#   """
#   This tool helps in adding questions that are unknown to a list, during the interview process.
#   """

#   print("Adding the question to unknown questions...")
#   uk_questions.append(question)

#   return "Question added to the list and will work on it"
  
# @tool
# def get_unknown_questions_list() -> list[str]:
#   """
#   This tool helps in retrieving the unknown questions.
#   """

#   print("Getting unknown questions...")
  
#   return uk_questions


@tool
def compose_email_llm(interviewer_details:list[str], my_details:list[str]) -> str:
  """This tool can create a draft email in a professional and formal tone and draft it to gmail."""

  print("Composing email....")

  email_compose_prompt = ChatPromptTemplate.from_messages(
    [
      ("system","You are an expert professional email generator and also a student actively seeking job opportunities. Compose email thanking the interviewer for connecting with you. In the email make sure to add my details like phone number, linkedin and github. And also once the email is ready use the available tools to draft that email in gmail."),
      ("user","Use the below interviewer details, my details and unknown questions to compose a professional email thanking the interviewer for connecting with you and proceeding with the further process. Make sure to add all my details in professional way possible. Always generate the mail in html. And never add anything about me or the interviewer like company or job role or name if it is not mentioned in their details. Once the email is ready draft it to gmail with the available tools.\n\nInterviewer Details:\n{interviewer_details}\n\nMy Details:\n{my_details}\n\nEmail:\n")
    ]
  )
  email_compose_query = email_compose_prompt.invoke({
    "interviewer_details":interviewer_details,
    "my_details": my_details})

  # print(email_compose_query)
  llm_to_compose_email = ChatOpenAI(temperature=1, api_key=openai_api_key)
  llm_to_compose_email_with_tools = llm_to_compose_email.bind_tools([compose_and_draft_mail])

  email_llm_response = llm_to_compose_email_with_tools.invoke(email_compose_query)
  if(email_llm_response.tool_calls):
      tool_response = exec_tools(llm_to_compose_email_with_tools, email_llm_response)

  # return "Successfully composed email and drafted."
  return "Mail drafted for Abhijeeth successfully. You will recieve the mail shortly."


llm = ChatOpenAI(temperature=1, model="gpt-4o", api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512, api_key=openai_api_key)
llm_with_tools = llm.bind_tools([compose_email_llm])
index_name = "alterego-index"

# pdf_loader = PyPDFLoader("Abhijeeth_Kollarapu_ds_ml_llm.pdf", )
# resume=pdf_loader.load()
# resume_content = resume[0].page_content

def load_md_resume(resume_md_file_path):
  with open(resume_md_file_path, 'rb') as f:
    md_resume_bytes = f.read()
  md_resume = md_resume_bytes.decode('utf-8')
  return md_resume



def load_and_embed_resume(url):
  # resume_loader = TextLoader(url)
  # resume_markdown = resume_loader.load()
  print("--------Started embedding---------------")
  md_resume = load_md_resume(url)
  resume_markdown = Document(page_content = md_resume, metadata={"source":"resume_markdown.md"})
  text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","name"), ("##","section")], strip_headers=False)
  resume_sections = text_splitter.split_text(resume_markdown.page_content)
  # vectorstore=FAISS.from_documents(embedding=embeddings, documents=resume_sections)

  # vectorstore = Chroma.from_documents(documents=resume_sections, embedding=embeddings, persist_directory="resume_vectorstore")
  # vectorstore.persist()
  # pc = Pinecone(api_key=pinecone_api_key)
  vector_store = PineconeVectorStore.from_documents(resume_sections, embedding=embeddings, index_name=index_name, pinecone_api_key=pinecone_api_key)

  print("--------Finished embedding---------------")

  
def get_retriever(vectorstore_url):
  # vectorstore = Chroma(embedding_function=embeddings, persist_directory=vectorstore_url)
  vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
  retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":4})
  return retriever

def ret_aug_gen(retriever, question, chat_history):
  
  retrieved_resume_slices = retriever.invoke(question)

  retrieved_resume_info = '\n'.join([info.page_content for info in retrieved_resume_slices])

  system_prompt = """You are Abhijeeth, a job-seeking professional currently attending a virtual job interview.\n
Your role is to impersonate Abhijeeth and respond professionally and concisely to interview questions.\n
You must only use information provided in the context extracted from Abhijeeth’s resume.\n
Most importantly try to provide the links/urls from the resume context and make sure it is clickable and navigates to correct page.
Do NOT make up or assume any experience or skills not explicitly present in the context.\n
Maintain a polite and confident tone in all your responses and try to keep the interview engaging by encouraging the interviewer to ask more.\n
\n
During the conversation, if it feels appropriate, ask the interviewer whether they would like to connect further.\n
If they agree, offer to share Abhijeeth’s contact details: phone number, email address, and LinkedIn profile.\n
Then ask the interviewer for their email address.\n
\n
If the interviewer provides their email, use the available tools to draft a professional thank-you or follow-up email addressed to them and make sure to include my details like phone, linkedin, github in that draft email\n
Save the email as a draft in Abhijeeth’s Gmail account. Do NOT send the email.\n
\n
You should only answer when the interviewer asks a question.\n
If they request elaboration, do so only within the boundaries of the context provided."""

  human_message="""
  Context from resume:\n
  {resume_context}\n
  \n
  Interviewer's Question:\n
  {interviewer_question}\n
  \n
  Based only on the above context, impersonate Abhijeeth and respond as he would in an interview.\n
  Answer briefly, professionally, and do NOT include any information not present in the resume context.\n
  \n
  If appropriate, ask the interviewer whether they would like to connect further.\n
  If they agree, offer to share Abhijeeth’s contact details (email, phone, and LinkedIn), and ask for their email address.\n
  \n
  If the interviewer provides their email, use the available tools to draft a professional Gmail message and save it as a draft.

  """


  prompt_template = ChatPromptTemplate.from_messages(
    messages=[
      ("system",system_prompt),
      MessagesPlaceholder("chat_history"),
      ("user",human_message)
    ],
  )

  
  prompt_to_llm = prompt_template.invoke({
      "chat_history": chat_history,
      "interviewer_question": question,
      "resume_context": retrieved_resume_info
  })

  response = llm_with_tools.invoke(prompt_to_llm)

  return response

def exec_tools(llm_new,ai_response):
  tool_name = globals().get(ai_response.tool_calls[0]["name"])
  tool_args = ai_response.tool_calls[0]["args"]
  tool_call_id = ai_response.tool_calls[0]["id"]

  tool_output = tool_name.invoke(tool_args)
  tool_response = ToolMessage(content=tool_output, tool_call_id=tool_call_id)
  return tool_response

  # ai_response_after_tool_exec=llm_new.invoke([ai_response, tool_response])

  # return ai_response_after_tool_exec, tool_response

def main():
  chat_history = []
  vectorstore_url="resume_vectorstore"
  retriever = get_retriever(vectorstore_url)
  while True:
    question=input("You (enter exit to exit): ")
    if(question.lower() == "exit"):
      break

    response = ret_aug_gen(retriever, question, chat_history)
    chat_history.append(HumanMessage(question))
    chat_history.append(response)
    if(response.tool_calls):
      # response, tool_response = exec_tools(llm_with_tools,response)
      tool_response = exec_tools(llm_with_tools,response)
      chat_history.append(tool_response)
      resp_after_tool_exec = llm_with_tools.invoke(chat_history)
      chat_history.append(resp_after_tool_exec)
    print("Ahijeeth Kollarapu:")
    print(chat_history[-1].content)


if __name__=="__main__":
  # print(llm_with_tools.__getattribute__("kwargs"))
  load_and_embed_resume("resume_markdown.md")
  # main()

  # print(embeddings)


  # compose_and_draft_mail.invoke({"sender":"abhijeethkollarapu@gmail.com", "reciepient":"abhijeeth1200@gmail.com", "subject":"This is being tested entirely", "message_text":"message_text"})

  # res=compose_email_llm.invoke({"interviewer_details":{"email":"abhijeeth1200@gmail.com"}, "my_details":{"email":"abhijeethkollarapu@gmail.com", "phone":"8134528290", "linkedin":"https://www.linkedin.com/in/abhijeeth-kollarapu", "github":"https://github.com/Abhijeeth8"}})

  # print(res.content)

