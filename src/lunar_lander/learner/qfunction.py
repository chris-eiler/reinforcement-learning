

# Importing requirements
import torch


def q_function(model, state, action, reward, next_state):
    q_values = model(state)
    q_value = q_values[action]
    next_q_values = model(next_state)
    next_q_value = torch.max(next_q_values)
    target = reward + 0.9 * next_q_value
    return (q_value - target) ** 2


def learn(agent, env, n_epochs: int = 10000):
    for epoch in range(n_epochs):
        state = env.reset()
        episode_over = False
        while not episode_over:
            action = agent.get_action(state)
            next_state, reward, episode_over, _, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
        agent.randomness = min(agent.min_randomness, agent.randomness * agent.randomness_decay)
    return agent

