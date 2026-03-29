from agent.agent import Agent

agent = Agent("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# Generation with System Prompt
# response = agent.generate_with_role("What is an AI agent?")

# Structured Generation with JSON output
schema = '''
{
    "explanation" : "string",
    "difficulty" : "beginner" | "intermediate" | "advanced"
}
'''

result = agent.generate_structured("Explain Computer Vision", schema)

print(result)