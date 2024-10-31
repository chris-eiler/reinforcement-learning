"""Reinforcement learning agent for the Lunar Lander environment.
"""


# Import requirements
import gymnasium as gym
from learner.agent import QLearner
from learner.qfunction import q_function, learn


SPACE = "LunarLander-v3"


def main():
    # Create the environment
    env = gym.make(SPACE)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create the agent
    agent = QLearner(n_states, n_actions)

    # Train the agent
    agent = learn(agent, env)

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
