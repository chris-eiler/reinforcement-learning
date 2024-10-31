"""Reinforcement learning agent for the Lunar Lander environment.
"""


# Import requirements
from torch import nn


class QLearner(nn.Module):
    def __init__(self, state_size, action_size):
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    pass
