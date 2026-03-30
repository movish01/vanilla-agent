from agent.agent import Agent

agent = Agent("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# # Generation with System Prompt
# response = agent.generate_with_role("What is an AI agent?")

# # Structured Generation with JSON output
# schema = '''
# {
#     "explanation" : "string",
#     "difficulty" : "beginner" | "intermediate" | "advanced"
# }
# '''

# # Decision making from model
# decision = agent.decide("what do you call hello in spanish", options=["answer_question", "summarize_text", "translate_text"])
# print(decision)

# # Tool request from model
# tool_call = agent.request_tool("What is 5 multiply by 7?")
# print(f"Tool request: {tool_call}")

# if tool_call:
#     result = agent.execute_tool_call(tool_call)
#     print(f"Tool result: {result}")

# # Agent Loop and states
# results = agent.run_loop("help me understand loops", max_steps=3)

# for i, result in enumerate(results, 1):
#     print(f"Step {i}")
#     action = result.get("action", "unknown")
#     reason = result.get("reason", "no reason provided")
#     print(f"Actions: {action}")
#     print(f"Reason: {reason}")
#     if i < len(results):
#         print()

# # Run agent with memory
# response_1 = agent.run_with_memory("My name is Amila, and I like white chocolate.")
# if response_1 and "reply" in response_1:
#     print(f"Response 1: {response_1['reply']}")

# response_2 = agent.run_with_memory("What's my name")
# if response_2 and "reply" in response_2:
#     print(f"Response 2: {response_2['reply']}")
    
# response_3 = agent.run_with_memory("I want something sweet, what should I eat")
# if response_3 and "reply" in response_3:
#     print(f"Respone 3: {response_3['reply']}")

# print(f"Memory contents: {agent.memory.get_all()}")

# # Agent with planning
# plan = agent.create_plan("Write a critique about movie A Beautiful Mind")
# print(f"Plan: {plan}")

# if plan:
#     results = agent.execute_plan(plan)
#     print(f"Execution results: {results}")

# # Atomic actions within Agent
# step = "Write a critique for movie A Beautiful Mind"
# atomic_action = agent.create_atomic_actions(step)
# print(f"Step: {step}")
# print(f"Atomic action: {atomic_action}")

# plan = agent.create_plan("Create a tutorial for python")
# if plan and "steps" in plan and plan["steps"]:
#     first_step = plan["steps"][0]
#     atomic_action_from_plan = agent.create_atomic_actions(first_step)
#     print(f"\nPlan step: {first_step}")
#     print(f"Atomic action from plan step: {atomic_action_from_plan}")

# # Atom of Thought - graph and execution
graph = agent.create_aot_plan("Research and write article on thriller movies last decade") 
print(f"AoT graph: {graph}")

if graph:
    results = agent.execute_aot_plan(graph)
    print(f"Execution results: {results}")