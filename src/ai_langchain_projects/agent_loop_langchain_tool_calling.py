from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product:str) -> float:
    """
    Look up the price of the product in the catalog
    """
    print(f">> Executing get_product_price (product: '{product}')")
    prices = {"laptop": 999.99, "phone": 499.99, "tablet": 299.99}
    return prices.get(product, 0)

@tool
def apply_discount(price:float, discount_tier: str) -> float:
    """
    Apply a discount tier to the price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f" >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)

#Agent Loop

@traceable(name="LangChain Agent Loop")
def run_agent(question:str):

    tools = [get_product_price, apply_discount]

    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")

    messages = [
        SystemMessage(
            content=("You are a helpful shopping assistant."
                   "You have access to product catalog tool"
                   "and a discount tool.\n\n"
                   "STRICT RULES - you must follow these exactly:\n"
                   "1. Never guess or assume any product price."
                   "You must call get_product_price first to get the real price.\n"
                   "2. Only call apply_discount after you received"
                   "price from the get_product_price. Pass the exact price"
                   "returned by get_product_price - do not pass a made up number.\n"
                   "3. Never calcuate discount yourself using math."
                   "Always use the apply_discount tool.\n"
                   "4. If the user does not specify the discount tier,"
                   "ask them which tier to use - do not assume one."
            )
        ),

        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tools_call = ai_message.tool_calls

        #if no tools call, this the final answer
        if not tools_call:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        #Process only the first tool call - force one tool per iteration
        tool_call = tools_call[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool returned] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict")

        observation = tool_to_use.invoke(tool_args)

        print(f" [Tool observation] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Error: Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")



