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
# 

results = agent.run_loop("help me understand loops", max_steps=3)

for i, result in enumerate(results, 1):
    print(f"Step {i}")
    action = result.get("action", "unknown")
    reason = result.get("reason", "no reason provided")
    print(f"Actions: {action}")
    print(f"Reason: {reason}")
    if i < len(results):
        print()
# print(decision)