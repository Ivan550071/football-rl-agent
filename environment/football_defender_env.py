"""Phase 2.3: one attacker versus a scripted defender (Gymnasium).

This is a NEW environment; it does not modify Phase 1 or use its Q-table.
Run from project root: python -m environment.football_defender_env
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FootballDefenderEnv(gym.Env):
    """Grid-based attacker vs. rule-based defender.

    Observation: [player_x, player_y, defender_x, defender_y, has_ball]
    Actions: 0 up, 1 down, 2 left, 3 right, 4 shoot.
    The defender moves toward the ball carrier every `defender_move_interval`
    steps. A collision while carrying the ball ends the episode (tackle).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    UP, DOWN, LEFT, RIGHT, SHOOT = range(5)
    ACTION_NAMES = ("up", "down", "left", "right", "shoot")
    MOVES = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

    def __init__(self, width=9, height=5, max_steps=80,
                 defender_move_interval=2, render_mode=None):
        super().__init__()
        if width < 7 or height < 3 or height % 2 == 0:
            raise ValueError("Use width >= 7 and an odd height >= 3")
        if max_steps < 1 or defender_move_interval < 1:
            raise ValueError("max_steps and defender_move_interval must be positive")
        if render_mode not in (None, "ansi"):
            raise ValueError("Only None or 'ansi' render_mode is supported")

        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.defender_move_interval = int(defender_move_interval)
        self.render_mode = render_mode
        self.n_actions = len(self.ACTION_NAMES)
        self.ball_pos = (2, height // 2)
        self.goal_pos = (width - 1, height // 2)
        self.shooting_spot = (width - 2, height // 2)

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.MultiDiscrete(
            [width, height, width, height, 2]
        )

        # Attacker starts near the ball; defender starts nearer the goal.
        self.attacker_starts = [
            (x, y) for x in range(3) for y in range(height)
            if (x, y) != self.ball_pos
        ]
        self.defender_starts = [
            (x, y) for x in range(width - 4, width - 1)
            for y in range(height)
            if (x, y) not in (self.goal_pos, self.ball_pos)
        ]
        self.player_pos = (0, height // 2)
        self.defender_pos = (width - 3, height // 2)
        self.has_ball = False
        self.steps = 0
        self.finished = False

    def _get_obs(self):
        return np.array([
            *self.player_pos, *self.defender_pos, int(self.has_ball)
        ], dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        player = options.get("player_pos")
        defender = options.get("defender_pos")
        if player is None:
            player = self.attacker_starts[
                int(self.np_random.integers(len(self.attacker_starts)))
            ]
        if defender is None:
            defender = self.defender_starts[
                int(self.np_random.integers(len(self.defender_starts)))
            ]
        player, defender = tuple(player), tuple(defender)
        if player not in self.attacker_starts or defender not in self.defender_starts:
            raise ValueError("Starting positions must be valid attacker/defender cells")
        if player == defender:
            raise ValueError("Attacker and defender cannot start on the same cell")

        self.player_pos = player
        self.defender_pos = defender
        self.has_ball = False
        self.steps = 0
        self.finished = False
        return self._get_obs(), {}

    def _defender_move(self):
        """One Manhattan step toward the ball carrier, with seeded tie-breaking."""
        dx = self.player_pos[0] - self.defender_pos[0]
        dy = self.player_pos[1] - self.defender_pos[1]
        candidates = []
        if dx != 0:
            candidates.append((self.defender_pos[0] + (1 if dx > 0 else -1),
                               self.defender_pos[1]))
        if dy != 0:
            candidates.append((self.defender_pos[0],
                               self.defender_pos[1] + (1 if dy > 0 else -1)))
        if candidates:
            self.defender_pos = candidates[
                int(self.np_random.integers(len(candidates)))
            ]

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode finished: call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        action = int(action)
        self.steps += 1
        reward = -0.1
        event = "moved"
        terminated = False

        if action in self.MOVES:
            dx, dy = self.MOVES[action]
            next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if not (0 <= next_pos[0] < self.width
                    and 0 <= next_pos[1] < self.height):
                reward -= 1.0
                event = "boundary"
            elif next_pos == self.defender_pos and not self.has_ball:
                reward -= 1.0
                event = "blocked"
            else:
                self.player_pos = next_pos
                if not self.has_ball and self.player_pos == self.ball_pos:
                    self.has_ball = True
                    reward += 5.0
                    event = "ball_collected"
        else:  # SHOOT
            if self.has_ball and self.player_pos == self.shooting_spot:
                reward += 100.0
                terminated = True
                event = "goal"
            else:
                reward -= 1.0
                event = "missed_shot"

        if not terminated and self.has_ball:
            # Defender can tackle immediately if attacker enters occupied cell.
            if (self.player_pos != self.defender_pos
                    and self.steps % self.defender_move_interval == 0):
                self._defender_move()
            if self.player_pos == self.defender_pos:
                self.has_ball = False
                reward -= 25.0
                terminated = True
                event = "tackled"

        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            event = "time_limit"
        self.finished = terminated or truncated

        return self._get_obs(), float(reward), terminated, truncated, {
            "event": event
        }

    def render(self):
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                position = (x, y)
                if position == self.player_pos:
                    symbol = "P"
                elif position == self.defender_pos:
                    symbol = "D"
                elif position == self.ball_pos and not self.has_ball:
                    symbol = "B"
                elif position == self.goal_pos:
                    symbol = "G"
                else:
                    symbol = "."
                row.append(symbol)
            rows.append(" ".join(row))
        return "\n".join(rows) + f"\nHas ball: {self.has_ball} | Steps: {self.steps}"

    def close(self):
        pass


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = FootballDefenderEnv()
    check_env(env, skip_render_check=True)
    print("Defender environment check passed!")
    obs, _ = env.reset(seed=42)
    print("Observation:", obs)
    print("Actions:", env.action_space)
    print(env.render())
    env.close()
