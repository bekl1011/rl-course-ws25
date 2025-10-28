import random
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)

print("## Frozen Lake ##")

no_states = env.observation_space.n
no_actions = env.action_space.n

q_values = np.zeros((no_states, no_actions))

alpha = 0.01


def play_episode(q_values):

    state = env.reset()[0]
    done = False
    r_s = []
    while not done:
        # choose greedy one of the best actions
        best_action_indices = np.where(q_values[state]
                                       == np.max(q_values[state]))[0]
        action = np.random.choice(best_action_indices)
        state, reward, done, _, _ = env.step(action)
        r_s.append(reward)
    return r_s


def learn_q_table():

    state = env.reset()[0]
    action = random.randint(0, 3)
    done = False

    r_s = []

    while not done:

        next_state, reward, done, _, _ = env.step(action)
        next_action = random.randint(0, 3)

        q_values[state, action] += (alpha*(reward
                                           + q_values[next_state, next_action]
                                           - q_values[state, action]))
        state = next_state
        action = next_action

        r_s.append(reward)

    return r_s


def main():
    successful_episodes = 1000
    plot_data = []
    while successful_episodes > 0:
        r_s = learn_q_table()

        if sum(r_s) > 0:
            # print(q_values)
            all_rewards = 0
            for _ in range(0, 100):
                rewards = play_episode(q_values)
                all_rewards += sum(rewards)
            avg_reward_sum = all_rewards / 100
            print(avg_reward_sum)
            plot_data.append(avg_reward_sum)
            successful_episodes -= 1

    # plot average rewards sum
    plt.figure()
    plt.xlabel("No. of random successfull episodes")
    plt.ylabel("Avg reward sum per episode over 100 greedy episodes")
    plt.plot(plot_data)
    plt.show()


main()
