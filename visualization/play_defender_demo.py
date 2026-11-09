"""Pygame demo for the Phase 2.3 defender environment.

Run from project root:
    python -m visualization.play_defender_demo
    python -m visualization.play_defender_demo --mode scripted

Modes: manual controls, or a simple rule-based demonstration bot.
The Phase 1 Q-table cannot control this environment (different observations).
"""

import argparse
from collections import deque

import numpy as np
import pygame

from environment.football_defender_env import FootballDefenderEnv
from environment.football_renderer import FootballRenderer


def scripted_action(env, rng):
    """Follow a shortest path around the defender. This is NOT RL."""
    if env.has_ball and env.player_pos == env.shooting_spot:
        return env.SHOOT

    target = env.shooting_spot if env.has_ball else env.ball_pos
    queue = deque([(env.player_pos, None)])
    visited = {env.player_pos}
    # Prefer goalward motion, but consider all four directions.
    directions = [env.RIGHT, env.UP, env.DOWN, env.LEFT]

    while queue:
        position, first_action = queue.popleft()
        if position == target:
            if first_action is not None:
                return first_action
            break
        for action in directions:
            dx, dy = env.MOVES[action]
            candidate = (position[0] + dx, position[1] + dy)
            if not (0 <= candidate[0] < env.width and 0 <= candidate[1] < env.height):
                continue
            if candidate in visited or candidate == env.defender_pos:
                continue
            # Avoid a square where the defender could tackle us this turn.
            if env.has_ball and (env.steps + 1) % env.defender_move_interval == 0:
                distance = (abs(candidate[0] - env.defender_pos[0])
                            + abs(candidate[1] - env.defender_pos[1]))
                if distance <= 1:
                    continue
            visited.add(candidate)
            queue.append((candidate, action if first_action is None else first_action))

    # When blocked, try a random legal move rather than freezing.
    possible = []
    for action, (dx, dy) in env.MOVES.items():
        dest = (env.player_pos[0] + dx, env.player_pos[1] + dy)
        if (0 <= dest[0] < env.width and 0 <= dest[1] < env.height
                and dest != env.defender_pos):
            possible.append(action)
    return int(rng.choice(possible)) if possible else env.SHOOT


def main():
    parser = argparse.ArgumentParser(description="Defender football demo")
    parser.add_argument("--mode", choices=("manual", "scripted"), default="manual")
    parser.add_argument("--delay", type=int, default=400,
                        help="Milliseconds between scripted actions")
    args = parser.parse_args()
    if args.delay < 60:
        parser.error("--delay must be at least 60 milliseconds")

    env = FootballDefenderEnv(max_steps=80)
    renderer = FootballRenderer(env.width, env.height)
    clock = pygame.time.Clock()
    rng = np.random.default_rng(42)

    keys = {
        pygame.K_UP: env.UP, pygame.K_w: env.UP,
        pygame.K_DOWN: env.DOWN, pygame.K_s: env.DOWN,
        pygame.K_LEFT: env.LEFT, pygame.K_a: env.LEFT,
        pygame.K_RIGHT: env.RIGHT, pygame.K_d: env.RIGHT,
        pygame.K_SPACE: env.SHOOT,
    }
    mode = args.mode
    observation, _ = env.reset(seed=42)
    reward_total = 0.0
    last_event = "ready"
    done = False
    last_bot_tick = pygame.time.get_ticks()
    finished_tick = 0

    def reset_episode():
        nonlocal observation, reward_total, last_event, done
        nonlocal last_bot_tick, finished_tick
        observation, _ = env.reset()
        reward_total = 0.0
        last_event = "ready"
        done = False
        last_bot_tick = pygame.time.get_ticks()
        finished_tick = 0

    def act(action):
        nonlocal observation, reward_total, last_event, done, finished_tick
        observation, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        last_event = info["event"]
        done = terminated or truncated
        if done:
            finished_tick = pygame.time.get_ticks()

    print("Manual: arrows/WASD move; SPACE shoot; TAB switch mode; R reset; ESC quit")
    print("Scripted mode uses a rule-based bot, NOT the Phase 1 Q-table.")
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
                        mode = "scripted" if mode == "manual" else "manual"
                        reset_episode()
                    elif mode == "manual" and not done:
                        action = keys.get(event.key)
                        if action is not None:
                            act(action)
            if not running:
                break

            now = pygame.time.get_ticks()
            if mode == "scripted":
                if done and now - finished_tick >= 1100:
                    reset_episode()
                elif not done and now - last_bot_tick >= args.delay:
                    act(scripted_action(env, rng))
                    last_bot_tick = now

            renderer.draw(env, mode=mode, total_reward=reward_total,
                          event=last_event, finished=done)
            clock.tick(60)
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
