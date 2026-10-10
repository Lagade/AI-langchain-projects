from .agent_loop_raw_function_calling import run_agent


def main() -> None:
    print("Hello LangChain Agent (raw function calling)!")
    print()
    run_agent("What is the price of a laptop with a gold discount?")
