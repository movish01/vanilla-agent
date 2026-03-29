from agent.agent import Agent

agent = Agent("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# Generation with System Prompt
# response = agent.generate_with_role("What is an AI agent?")

# Structured Generation with JSON output
# schema = '''
# {
#     "explanation" : "string",
#     "difficulty" : "beginner" | "intermediate" | "advanced"
# }
# '''

# Decision making from model
# decision = agent.decide("what do you call hello in spanish", options=["answer_question", "summarize_text", "translate_text"])

# Tool request from model
tool_call = agent.request_tool("What is 5 multiply by 7?")
print(f"Tool request: {tool_call}")

if tool_call:
    result = agent.execute_tool_call(tool_call)
    print(f"Tool result: {result}")

# print(decision)