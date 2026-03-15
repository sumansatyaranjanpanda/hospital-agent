from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict

from langgraph.graph.message import add_messages

from prompt_library.prompt import booking_prompt, information_prompt, response_prompt, system_prompt
from toolkit.toolkits import (
    cancel_appointment,
    check_availability_by_doctor,
    check_availability_by_specialization,
    find_doctors_by_specialization,
    list_all_specializations,
    recommend_doctor_for_query,
    reschedule_appointment,
    set_appointment,
)
from utils.llms import LLMModel


class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    id_number: int
    next: str
    query: str
    current_reasoning: str
    worker_name: str
    worker_summary: str


class DoctorAppointmentAgent:
    def __init__(self):
        llm_model = LLMModel()
        self.llm_model = llm_model.get_model()

    def supervisor_node(self, state: AgentState) -> Command[Literal["information_node", "booking_node", "__end__"]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Patient ID: {state['id_number']}\nLatest user request: {state['messages'][-1].content}",
            },
        ]

        if len(state["messages"]) > 1:
            history = "\n".join(
                [f"{message.type.upper()}: {message.content}" for message in state["messages"][:-1][-6:]]
            )
            messages.append({"role": "user", "content": f"Recent conversation context:\n{history}"})

        response = self.llm_model.with_structured_output(Router).invoke(messages)
        goto = END if response["next"] == "FINISH" else response["next"]
        latest_query = state["messages"][-1].content if state["messages"] else state.get("query", "")

        return Command(
            goto=goto,
            update={
                "next": response["next"],
                "query": latest_query,
                "current_reasoning": response["reasoning"],
            },
        )

    def _run_worker(self, worker_prompt: str, tools: list[Any], state: AgentState, worker_name: str) -> Command[Literal["response_node"]]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", worker_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        worker_agent = create_react_agent(model=self.llm_model, tools=tools, prompt=prompt)
        worker_messages = [
            HumanMessage(
                content=(
                    f"System context: patient hospital ID is {state['id_number']}. "
                    "Use this ID for booking tools and do not ask the user for it again."
                )
            )
        ] + state["messages"]
        result = worker_agent.invoke({"messages": worker_messages})
        worker_summary = result["messages"][-1].content

        return Command(
            goto="response_node",
            update={
                "worker_name": worker_name,
                "worker_summary": worker_summary,
            },
        )

    def information_node(self, state: AgentState) -> Command[Literal["response_node"]]:
        return self._run_worker(
            information_prompt,
            [
                check_availability_by_doctor,
                check_availability_by_specialization,
                find_doctors_by_specialization,
                list_all_specializations,
                recommend_doctor_for_query,
            ],
            state,
            "information_node",
        )

    def booking_node(self, state: AgentState) -> Command[Literal["response_node"]]:
        return self._run_worker(
            booking_prompt,
            [set_appointment, cancel_appointment, reschedule_appointment],
            state,
            "booking_node",
        )

    def response_node(self, state: AgentState) -> Command[Literal["__end__"]]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", response_prompt),
                (
                    "user",
                    "Patient ID: {id_number}\n"
                    "Latest user request: {query}\n"
                    "Chosen worker: {worker_name}\n"
                    "Worker reasoning: {current_reasoning}\n"
                    "Worker result:\n{worker_summary}",
                ),
            ]
        )

        response = self.llm_model.invoke(
            prompt.format_messages(
                id_number=state["id_number"],
                query=state["query"],
                worker_name=state.get("worker_name", ""),
                current_reasoning=state.get("current_reasoning", ""),
                worker_summary=state.get("worker_summary", ""),
            )
        )

        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=response.content)],
            },
        )

    def workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("information_node", self.information_node)
        graph.add_node("booking_node", self.booking_node)
        graph.add_node("response_node", self.response_node)
        graph.add_edge(START, "supervisor")
        self.app = graph.compile()
        return self.app
