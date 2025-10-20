import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)

print("## Frozen Lake ##")

no_states = env.observation_space.n
no_actions = env.action_space.n

q_values = np.zeros((no_states, no_actions))
q_counter = np.zeros((no_states, no_actions))


def play_episode(q_values=None):

    state = env.reset()[0]
    done = False
    r_s = []
    s_a = []
    while not done:
        if q_values is None:
            action = random.randint(0, 3)
        else:
            # choose greedy one of the best actions
            best_action_indices = np.where(q_values[state]
                                           == np.max(q_values[state]))[0]
            action = np.random.choice(best_action_indices)

        s_a.append((state, action))
        state, reward, done, _, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s


def main():
    successful_episodes = 1000
    plot_data = []
    while successful_episodes > 0:
        # play random episode
        s_a, r_s = play_episode()

        # update q-values with MC-prediction
        for i, (s, a) in enumerate(s_a):
            return_i = sum(r_s[i:])
            q_counter[s][a] += 1
            q_values[s][a] += 1/q_counter[s][a] * (return_i - q_values[s][a])

        # get mean return for greedy policy
        if sum(r_s) > 0:
            # average reward sum over 100 episodes
            all_rewards = 0
            for i in range(0, 100):
                s_a, rewards = play_episode(q_values)
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
