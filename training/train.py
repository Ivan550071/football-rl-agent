"""Train a tabular Q-learning agent on the grid-based football environment.

Run from project root:
    python -m training.train
"""

from pathlib import Path
import numpy as np

from environment.football_env import FootballEnv
from agents.q_learning import QLearningAgent

EPISODES = 20_000
REPORT_EVERY = 1_000
SEED = 42


def train():
    np.random.seed(SEED)
    env = FootballEnv()
    agent = QLearningAgent(
        width=env.width,
        height=env.height,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9997,
        epsilon_min=0.05,
    )

    rewards = []
    goals = []

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        scored = False

        for _ in range(env.max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if info["event"] == "goal":
                scored = True
            if done:
                break

        agent.decay_exploration()
        rewards.append(episode_reward)
        goals.append(int(scored))

        if episode % REPORT_EVERY == 0:
            avg_reward = np.mean(rewards[-REPORT_EVERY:])
            goal_rate = 100 * np.mean(goals[-REPORT_EVERY:])
            print(f"Episode {episode:>6}/{EPISODES} | "
                  f"avg reward: {avg_reward:>7.2f} | "
                  f"goal rate: {goal_rate:5.1f}% | "
                  f"epsilon: {agent.epsilon:.3f}")

    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "q_table.npy"
    np.save(model_path, agent.q_table)
    print(f"\nSaved Q-table to: {model_path}")

    return agent


if __name__ == "__main__":
    trained_agent = train()
    np.random.seed(123)
    env = FootballEnv()
    successes = 0
    for _ in range(100):
        state, _ = env.reset()
        for _ in range(env.max_steps):
            action = trained_agent.choose_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated:
                successes += 1
            if terminated or truncated:
                break
    print(f"Greedy evaluation: {successes}% goals over 100 runs")
