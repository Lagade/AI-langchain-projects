from .agent_loop_langchain_tool_calling import run_agent


def main() -> None:
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    run_agent("What is the price of a laptop with a gold discount?")
