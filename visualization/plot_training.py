import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "training_metrics.csv"

WINDOW = 200


def rolling_mean(values, window):
    """Calculate moving average over recent episodes."""
    values = np.asarray(values, dtype=float)
    cumulative = np.r_[0.0, np.cumsum(values)]

    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)

    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def plot_line(x, y, title, ylabel, filename, ylim=None):
    """Generate and save a training plot."""
    plt.figure(figsize=(10, 5))

    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=160)
    plt.close()

    print(f"Saved: {filename}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "Training data not found! Run training first."
        )

    # Read training statistics
    with open(CSV_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Training CSV is empty.")

    episodes = np.array([int(row["episode"]) for row in rows])
    rewards = np.array([float(row["reward"]) for row in rows])
    goals = np.array([int(row["goal"]) for row in rows])
    epsilons = np.array([float(row["epsilon"]) for row in rows])

    # Plot 1: Average reward
    plot_line(
        episodes,
        rolling_mean(rewards, WINDOW),
        "Q-learning: Average Reward",
        "Average Reward",
        "average_reward.png"
    )

    # Plot 2: Goal scoring rate
    plot_line(
        episodes,
        100 * rolling_mean(goals, WINDOW),
        "Q-learning: Goal Scoring Rate",
        "Goal Rate (%)",
        "goal_rate.png",
        (0, 105)
    )

    # Plot 3: Epsilon decay
    plot_line(
        episodes,
        epsilons,
        "Q-learning: Epsilon Decay",
        "Epsilon",
        "epsilon_decay.png",
        (0, 1.05)
    )


if __name__ == "__main__":
    main()
