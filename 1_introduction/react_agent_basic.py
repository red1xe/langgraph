from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import TavilySearchResults

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

search_tool = TavilySearchResults(serch_depth="basic")

agent = create_agent(
    llm,
    tools=[search_tool],
    system_prompt="Run all avaible tools without waiting approval",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Give me a funny tweet about todat's weather in Kayseri",
            }
        ]
    }
)

print(result)
