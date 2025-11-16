from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from chains import generation_chain, reflection_chain
from typing_extensions import TypedDict

load_dotenv()


class State(TypedDict):
    messages: List[BaseMessage]


graph = StateGraph(State)

REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state: State):
    new_message = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [new_message]}


def reflect_node(state: State):
    critique_message = reflection_chain.invoke({"messages": state["messages"]})
    feedback_message = HumanMessage(content=critique_message.content)
    return {"messages": state["messages"] + [feedback_message]}


def should_continue_reflection(state: State):
    if len(state["messages"]) > 4:
        return False
    return True


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)
graph.add_conditional_edges(
    GENERATE, should_continue_reflection, {True: REFLECT, False: END}
)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(
    {"messages": [HumanMessage(content="Write a twitter post about AI in healthcare.")]}
)
print(response)
