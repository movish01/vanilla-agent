from agent.agent import Agent

agent = Agent("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
response = agent.generate_with_role("What is an AI agent?")
print(response)