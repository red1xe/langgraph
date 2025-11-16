from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
generation_promt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie infulencer assistamt tasked with writing exellent twitter posts."
            "Generate the best twitter posts possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attemsts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate recommendations to improve the tweet."
            "Always provide specific suggestions for improvement, including tone, style, and content etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

generation_chain = generation_promt | llm
reflection_chain = reflection_prompt | llm
