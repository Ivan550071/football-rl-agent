"""Manual football game / Q-learning replay in a Pygame window.

Run from project root:
    python -m visualization.play_demo
    python -m visualization.play_demo --mode ai

Manual: arrows/WASD move, SPACE shoot. TAB switches manual/AI when a model
exists; R resets, ESC quits. No training or model updates take place here.
"""

import argparse
from pathlib import Path

import numpy as np
import pygame

from environment.football_gym_env import FootballGymEnv
from environment.football_renderer import FootballRenderer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q_TABLE_PATH = PROJECT_ROOT / "models" / "q_table.npy"


def load_q_table(env):
    """Load the Phase 1 model, or return None for manual-only play."""
    if not Q_TABLE_PATH.exists():
        return None

    table = np.load(Q_TABLE_PATH, allow_pickle=False)
    expected = (
        env.game.width, env.game.height, 2, env.game.n_actions
    )
    if table.shape != expected:
        raise ValueError(
            f"Saved Q-table shape {table.shape} != expected {expected}. "
            "Retrain Phase 1 before running the AI demo."
        )
    return table


def choose_ai_action(observation, q_table, rng):
    """Greedy action; random tie-breaking, with NO exploration or updates."""
    state = tuple(int(value) for value in observation)
    q_values = q_table[state]
    actions = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(actions))


def main():
    parser = argparse.ArgumentParser(description="Football RL Pygame demo")
    parser.add_argument(
        "--mode", choices=("manual", "ai"), default="manual",
        help="Start with keyboard controls or the trained Q-learning agent",
    )
    parser.add_argument(
        "--delay", type=int, default=420,
        help="Milliseconds between AI actions (default: 420)",
    )
    args = parser.parse_args()
    if args.delay < 50:
        parser.error("--delay must be at least 50 milliseconds")

    # Keep graphics separate from the environment: no rendering on every step.
    env = FootballGymEnv(max_steps=60, random_start=True, render_mode=None)
    q_table = load_q_table(env)
    if args.mode == "ai" and q_table is None:
        raise FileNotFoundError(
            f"No trained model found at {Q_TABLE_PATH}. "
            "Run 'python -m training.train' first, or start in manual mode."
        )

    renderer = FootballRenderer(env.game.width, env.game.height)
    clock = pygame.time.Clock()
    rng = np.random.default_rng(42)
    keys_to_actions = {
        pygame.K_UP: env.game.UP, pygame.K_w: env.game.UP,
        pygame.K_DOWN: env.game.DOWN, pygame.K_s: env.game.DOWN,
        pygame.K_LEFT: env.game.LEFT, pygame.K_a: env.game.LEFT,
        pygame.K_RIGHT: env.game.RIGHT, pygame.K_d: env.game.RIGHT,
        pygame.K_SPACE: env.game.SHOOT,
    }
    mode = args.mode
    observation, _ = env.reset(seed=42)
    total_reward = 0.0
    last_event = "ready"
    done = False
    last_ai_tick = pygame.time.get_ticks()
    end_tick = 0

    print("Controls: arrows/WASD move, SPACE shoot, TAB switch mode, R reset, ESC quit")
    if q_table is None:
        print("No models/q_table.npy found; AI mode unavailable, manual mode works.")

    def reset_episode():
        nonlocal observation, total_reward, last_event, done, end_tick, last_ai_tick
        observation, _ = env.reset()
        total_reward = 0.0
        last_event = "ready"
        done = False
        end_tick = 0
        last_ai_tick = pygame.time.get_ticks()

    def apply_action(action):
        nonlocal observation, total_reward, last_event, done, end_tick
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        last_event = info.get("event", "moved")
        done = terminated or truncated
        if done:
            end_tick = pygame.time.get_ticks()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_episode()
                    elif event.key == pygame.K_TAB:
                        if mode == "manual" and q_table is None:
                            print("AI mode requires models/q_table.npy")
                        else:
                            mode = "ai" if mode == "manual" else "manual"
                            reset_episode()
                    elif mode == "manual" and not done:
                        action = keys_to_actions.get(event.key)
                        if action is not None:
                            apply_action(action)

            if not running:
                break

            now = pygame.time.get_ticks()
            if mode == "ai":
                if done and now - end_tick >= 1300:
                    reset_episode()
                elif not done and now - last_ai_tick >= args.delay:
                    action = choose_ai_action(observation, q_table, rng)
                    apply_action(action)
                    last_ai_tick = now

            renderer.draw(
                env.game, mode=mode, total_reward=total_reward,
                event=last_event, finished=done,
            )
            clock.tick(60)
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
