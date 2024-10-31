"""Reinforcement learning agent for the Frozen Lake environment.
"""


# Import requirements
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class QLearner:
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.9, discount_factor: float = 0.9, min_randomness: float = 0.001):
        self.q_table = np.zeros((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.randomness = 3.0
        self.randomness_decay = 0.9999
        self.min_randomness = min_randomness
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, state: int) -> int:
        rand_num = np.random.uniform()
        if rand_num < self.randomness:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state, :])

    def update_q_table(self, state: int, action: int, reward: float, next_state: int) -> None:
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
        )

    def learn(self, env, n_epochs: int = 30_000) -> None:
        for _ in tqdm(range(n_epochs)):
            state, _ = env.reset()
            episode_over = False
            terminated = False
            ii = 0
            self.randomness = max(self.min_randomness, self.randomness * self.randomness_decay)
            while not episode_over and not terminated:
                action = self.get_action(state)
                next_state, reward, episode_over, terminated, _ = env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                ii += 1
                if ii > 45:
                    terminated = True


def play_game(agent):
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True, map_name="8x8")
    state, _ = env.reset()
    env.render()
    episode_over = False
    while not episode_over:
        action = agent.get_action(state)
        next_state, reward, episode_over, _, _ = env.step(action)
        state = next_state
    # env.close()

def main():
    # Create the environment
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Create the agent
    agent = QLearner(n_states, n_actions)

    # Train the agent
    agent.learn(env)

    # Close the environment
    env.close()

    # Play a game
    play_game(agent)


if __name__ == "__main__":
    agent = main()
