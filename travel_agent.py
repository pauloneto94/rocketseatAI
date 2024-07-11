import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import bs4


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-3.5-turbo")

# query = """
#     Vou viajar para Londres em agosto de 2024.
#     Quero um roteiro para viagem para mim com eventos que irão ocorrer na data da viagem e com o preço de passagem de Curitiba para Londres 
# """

def research_agent(llm, query):
    tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt)
    web_context = agent_executor.invoke({"input": query})

    return web_context['output']

def load_data():
    loader = WebBaseLoader(
        web_paths=("https://www.dicasdeviagem.com/inglaterra/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading background-imaged loading-dark")))
    )
    docs = loader.load()
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_spliter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever

def get_relevant_docs(query):
    retriever = load_data()
    relevant_documents = retriever.invoke(query)

    return relevant_documents

def supervisor_agent(llm, query, web_context, relevant_documents):
    prompt_template = """
        Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado.
        Utilize o contexto de eventos e preços de passagens, o input do usuário e os documentos relevantes para elaborar o roteiro.
        Context: {web_context}
        Documento relevante: {relevant_documents}
        Usuário: {query}
        Assistente:
    """

    prompt = PromptTemplate(
        input_variables=['web_context', 'relevant_documents', 'query'],
        template=prompt_template
    )

    sequence = RunnableSequence(prompt | llm)

    response = sequence.invoke({
        "web_context": web_context,
        "relevant_documents": relevant_documents,
        "query": query
    })

    return response

def get_response(llm, query):
    web_context = research_agent(llm=llm, query=query)
    relevant_documents = get_relevant_docs(query=query)
    response = supervisor_agent(llm=llm, query=query, web_context=web_context, relevant_documents=relevant_documents)

    return response

def lambda_handler(event, context):
    query = event.get("question")
    response = get_response(llm=llm, query=query)

    return {"body": response.content, "status": 200}