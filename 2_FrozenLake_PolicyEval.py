import gymnasium as gym
import random
from rich.console import Console

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
discount = 1

random.seed(0)

actions = range(0, env.unwrapped.action_space.n)
state_size = env.unwrapped.nrow * env.unwrapped.ncol
states = range(0, state_size)

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
state = env.reset(seed=0)

# here is an example how to access the transitions in the MDP
tp_matrix = env.unwrapped.P
state=0
action=1
print(f"Transition probabilities from state {state} with action {action}:")
for p, s_next, reward, _ in tp_matrix[state][action]:
	print("Probability", p)
	print("Next state", s_next)
	print("Reward", reward)

# here you can see the whole matrix in pretty print
c = Console()
c.print(env.unwrapped.P)

# your solution comes here

