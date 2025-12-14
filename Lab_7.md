# Lab 7

## Task 1

Check the file [`7_PolicyBased_CEM.ipynb`](7_PolicyBased_CEM.ipynb)
which implements the cross-entropy method (CEM) on the [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment.

This implementation is incomplete (and does not work). To fix it, implement the following:

1. Currently the state-action pairs of all episodes of one batch are used for training. To implement the Cross-Entropy-Method (CEM) correctly, use only the state-action pairs of the 20 best episodes in one batch in terms of highest reward.
2. Train the network with the correction from step 1.
3. After training, test the agent by running one episode using the trained network and record the episode
(see the notebook from last week for code templates for recording.)

## Task 2
Look through the notebook [`7_PolicyBased_Actor-Critic.ipynb`](7_PolicyBased_Actor-Critic.ipynb) and execute it to train the actor-critic 
agent.

Try to understand the code  and answer the following questions:

1. Why does the policy network has a softmax output (and the value network not?)
2. Why does the `select_action` method in cell 5 return two values?
3. Why do we need the `unsqueeze(0)` in the line `state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)` from cell 5
4. Why is there a minus in `policy_loss = -log_prob * td_error` in cell 8.
5. Is there any exploration in the agent's action choice? If yes, in which code line?

## Bonus: Task 3
The approach in notebook [`7_PolicyBased_Actor-Critic.ipynb`](7_PolicyBased_Actor-Critic.ipynb) uses two seperate networks for the actor and the critic. Change the code in such a way that we have only one network for both (like seen in the lecture).