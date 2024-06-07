import os
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import load_tools
from crewai import Agent, Task, Process, Crew
from langchain.llms import Ollama
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)
import streamlit as st

load_dotenv('../.env')
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
aihubmix_api_key = os.getenv("AIHUBMIX_API_KEY")  # AIHUBMIX

#print(serper_api_key)
#print(openai_api_key)
#print(aihubmix_api_key)

os.environ["SERPER_API_KEY"] = serper_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# 1. Tool for search

#search = GoogleSerperAPIWrapper()

#search_tool = Tool(
#    name="Scrape google searches",
#    func=search.run,
#    description="useful for when you need to ask the agent to search the internet",
#)

search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()


# Loading Human Tools
human_tools = load_tools(["human"])

#To Load LLAMA3
#localllm = Ollama(model="llama3")
#localllm = Ollama(model="llama3" ,verbose=False, base_url="http://192.168.10.158:11434")
localllm = Ollama(model="llama3" ,verbose=False, base_url="http://192.168.10.107:11434")
#localllm = Ollama(model="llama3:70b-instruct", verbose=False, base_url="http://192.168.10.158:11434")  
#localllm = Ollama(model="openchat")
#localllm = Ollama(model="llama3", verbose=False, base_url="http://host.docker.internal:11434")
#llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
#llm = ChatOpenAI(
#    openai_api_base="https://aihubmix.com/v1", # 注意，末尾要加 /v1
#    openai_api_key=aihubmix_api_key,
#    model="gpt-3.5-turbo-16k-0613"
#)

# Get your crew to work!
def kickoff_crew(query):
    
    print("start kickoff crew")
    print(query)

    input_query = query
    input_researcher_goal = "Find and explore the most exciting projects and companies in {}".format(input_query)
    input_researcher_backstory = "You are and Expert strategist that knows how to spot emerging trends and companies in {}. You're great at finding interesting, exciting projects on LocalLLama subreddit. You turned scraped data into detailed reports with names of most exciting projects an companies. ONLY use scraped data from the internet for the report.".format(input_query)
    input_writer_goal = "Write engaging and interesting blog post about {} using simple, layman vocabulary".format(input_query)
    input_writer_backstory = "You are an Expert Writer on technical innovation, especially in the field of {}. You know how to write in engaging, interesting but simple, straightforward and concise. You know how to present complicated technical terms to general audience in a fun way by using layman words.ONLY use scraped data from the internet for the blog.".format(input_query)
    input_critic_goal = "Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple and concise"
    input_critic_backstory = "You are an Expert at providing feedback to the technical writers. You can tell when a blog text isn't concise, simple or engaging enough. You know how to provide helpful feedback that can improve any text. You know how to make sure that text stays technical and insightful by using layman terms"

    company_researcher = Agent(
        role="Senior Researcher",
        goal=input_researcher_goal,
        backstory=input_researcher_backstory,
        verbose=True,
        allow_delegation=False,
    #    tools=[search_tool],
        tools=[search_tool, web_rag_tool],    
        llm=localllm,  # remove to use default gpt-4    
    )

    writer = Agent(
        role="Senior Technical Writer",
        goal=input_writer_goal,
        backstory=input_writer_backstory,
        verbose=True,
        allow_delegation=True,
        llm=localllm,  # remove to use default gpt-4       
    )
    critic = Agent(
        role="Expert Writing Critic",
        goal=input_critic_goal,
        backstory=input_critic_backstory,
        verbose=True,
        allow_delegation=True,
        llm=localllm,  # remove to use default gpt-4    
)

    task_report = Task(
        description="""Use and summarize scraped data from the internet to make a detailed report on the latest rising project ideas. Use ONLY 
        scraped data to generate the report. Your final answer MUST be a full analysis report, text only, ignore any code or anything that 
        isn't text. The report has to have bullet points and with 5-10 exciting new projects and tools. Write names of every tool and project. 
        Each bullet point MUST contain 8 sentences that refer to one specific company, product, model, URL link of the source or anything you found on the internet.  
        """,
        expected_output="Full analysis report in bullet points with the URL link of the source",
        agent=company_researcher,
    )


    task_blog = Task(
        description="""Write a blog article with text only and with a short but impactful headline and at least 10 paragraphs. Blog should summarize 
        the report on latest ai tools found on localLLama subreddit. Style and tone should be compelling and concise, fun, technical but also use 
        layman words for the general public. Name specific new, exciting projects, apps and companies. Don't 
        write "**Paragraph [number of the paragraph]:**", instead start the new paragraph in a new line. Write names of projects and tools in BOLD.
        ALWAYS include links to projects/tools/research papers.
        For your Outputs use the following markdown format:
        ```
        ## [Title of post](URL link of the project source)
        - Interesting facts
        - Own thoughts on how it connects to the overall theme of the newsletter
        ## [Title of second post](URL link of the project source)
        - Interesting facts
        - Own thoughts on how it connects to the overall theme of the newsletter
        ```
        """,
        expected_output="Article with impactful headline and at least 10 paragraphs",
        agent=writer,
    )

    task_critique = Task(
        description="""The Output MUST have the following markdown format:
        ```
        ## [Title of post](URL link of the project source)
        - Interesting facts
        - Own thoughts on how it connects to the overall theme of the newsletter
        ## [Title of second post](URL link of the project source)
        - Interesting facts
        - Own thoughts on how it connects to the overall theme of the newsletter
        ```
        Make sure that it does and if it doesn't, rewrite it accordingly.
        """,
        expected_output="Text with markdown format",
        agent=critic,
    )

    # instantiate crew of agents
    # crew = Crew(
    #     agents=[company_researcher, writer, critic],
    #     tasks=[task_report, task_blog, task_critique],
    #     verbose=2,
    #     process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    # )

    crew = Crew(
        agents=[company_researcher, writer],
        tasks=[task_report, task_blog],
        verbose=2,
        process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )

    result = crew.kickoff()
    return(result)


# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="迷途小書僮 The Little Lost Scholar Companion", page_icon=":boy:")

    st.header("迷途小書僮 The Little Lost Scholar Companion :boy:")
    query = st.text_input("Research goal")
    
    if query:
        st.write("Doing company research for ", query)

        result = kickoff_crew(query)

        st.info(result)

if __name__ == '__main__':
    #main()
    result = kickoff_crew("Place to visit in United Kingdom in 2024")
    print("迷途小書僮 The Little Lost Scholar Companion :boy:")
    print(result)