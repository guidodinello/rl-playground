from __future__ import annotations

import gymnasium as gym
from tqdm import tqdm

from truco.agent import QlearningAgent

gym.pprint_registry()

env = gym.make("truco/TrucoUruguayo-v0")

# hyperparameters
learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = QlearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

env.close()

# plot_value_policy(agent, usable_ace=False)
# plot_value_policy(agent, usable_ace=True)
# plot_training_error(agent, env)

# watch the agent play
# obs, info = env.reset()
# done = False
# while not done:
#     action = agent.get_action(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()

# env.close()
# env.close()
# env.close()
