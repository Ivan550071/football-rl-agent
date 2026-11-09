"""Evaluate the saved tabular Q-learning policy against a random baseline.

Run from the project root:
    python -m evaluation.evaluate

Train first if models/q_table.npy does not exist:
    python -m training.train
"""

from pathlib import Path
import numpy as np

from agents.q_learning import QLearningAgent
from environment.football_env import FootballEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "q_table.npy"
NUM_EPISODES = 1_000
SEED = 123


def load_agent(env):
    """Load learned Q-values into an agent; never train during evaluation."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No trained Q-table found at {MODEL_PATH}\n"
            "Run 'python -m training.train' from the project root first."
        )

    agent = QLearningAgent(
        width=env.width,
        height=env.height,
        n_actions=env.n_actions,
        epsilon=0.0,
    )
    saved_q_table = np.load(MODEL_PATH, allow_pickle=False)
    if saved_q_table.shape != agent.q_table.shape:
        raise ValueError(
            f"Q-table shape {saved_q_table.shape} doesn't match "
            f"the environment's expected {agent.q_table.shape}."
        )
    agent.q_table = saved_q_table
    return agent


def evaluate_policy(env, choose_action, episodes):
    """Evaluate a policy WITHOUT updating its Q-values.

    Returns goal count, average reward, average episode length, and
    average length of successful episodes.
    """
    goals = 0
    total_rewards = []
    episode_lengths = []
    successful_lengths = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for step_number in range(1, env.max_steps + 1):
            action = int(choose_action(state))
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if info["event"] == "goal":
                goals += 1
                successful_lengths.append(step_number)

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        episode_lengths.append(step_number)

    return {
        "goals": goals,
        "goal_rate": 100 * goals / episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_steps": float(np.mean(episode_lengths)),
        "mean_steps_to_goal": (
            float(np.mean(successful_lengths)) if successful_lengths else None
        ),
    }


def print_results(label, results, episodes):
    print(f"\n{label} ({episodes} episodes)")
    print(f"  Goals scored:      {results['goals']}/{episodes}")
    print(f"  Goal rate:         {results['goal_rate']:.1f}%")
    print(f"  Average reward:    {results['mean_reward']:.2f}")
    print(f"  Average steps:     {results['mean_steps']:.2f}")
    steps_to_goal = results["mean_steps_to_goal"]
    if steps_to_goal is None:
        print("  Steps per goal:    N/A (no goals scored)")
    else:
        print(f"  Steps per goal:    {steps_to_goal:.2f}")


def show_example_match(env, agent):
    """Render one match controlled entirely by the trained policy."""
    state, _ = env.reset()
    total_reward = 0.0
    print("\n=== Example match: trained agent ===")
    print("Starting state:", state)
    print(env.render())

    for step in range(1, env.max_steps + 1):
        action = agent.choose_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"\nStep {step} | Action: {env.ACTION_NAMES[action]} | "
            f"Reward: {reward:+.1f} | Event: {info['event']}"
        )
        print(env.render())
        if terminated or truncated:
            break

    print(f"\nResult: {info['event']} | Total reward: {total_reward:.1f}")


def main():
    env = FootballEnv()
    agent = load_agent(env)

    # Fix random seeds so both evaluations can be reproduced.
    np.random.seed(SEED)
    trained = evaluate_policy(
        env,
        lambda state: agent.choose_action(state, training=False),
        NUM_EPISODES,
    )

    random_rng = np.random.default_rng(SEED)
    random = evaluate_policy(
        env,
        lambda state: random_rng.integers(env.n_actions),
        NUM_EPISODES,
    )

    print(f"Loaded trained Q-table: {MODEL_PATH}")
    print_results("Trained Q-learning agent", trained, NUM_EPISODES)
    print_results("Random-action baseline", random, NUM_EPISODES)

    show_example_match(env, agent)

    print("\nNote: every episode uses the same starting configuration.")
    print("This test measures performance on the trained task, not generalization.")


if __name__ == "__main__":
    main()
