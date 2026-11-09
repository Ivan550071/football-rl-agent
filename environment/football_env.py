"""Phase 1: A tiny grid-based football environment for tabular Q-learning.

No third-party packages required. Its reset/step signatures follow the
Gymnasium convention, but this class is not yet a Gymnasium Env subclass.
"""
import random

class FootballEnv:
    # Actions are integers so they can later index a Q-table.
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SHOOT = 4

    ACTION_NAMES = ("up", "down", "left", "right", "shoot")
    MOVES = {
        UP: (0, -1),
        DOWN: (0, 1),
        LEFT: (-1, 0),
        RIGHT: (1, 0),
    }

    def __init__(self, max_steps=60, random_start=False, seed=None):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.width = 7
        self.height = 5
        self.start_pos = (0, 2)
        self.ball_pos = (2, 2)
        self.goal_pos = (6, 2)
        self.max_steps = max_steps
        self.n_actions = len(self.ACTION_NAMES)

        #Random starting positions
        self.random_start = random_start
        self.rng = random.Random(seed)

        self.start_positions = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in (self.ball_pos, self.goal_pos)
        ]

        self.reset()

    def _get_state(self):
        """(player_x, player_y, has_ball), usable as a dictionary key."""
        x, y = self.player_pos
        return (x, y, int(self.has_ball))

    def reset(self):
        """Start a new episode. Returns (state, info)."""
        
        if self.random_start:
            self.player_pos = self.rng.choice(self.start_positions)
        else:
            self.player_pos = self.start_pos
        
        self.has_ball = False
        self.steps = 0
        self.finished = False
        return self._get_state(), {}

    def step(self, action):
        """Run one turn: returns (state, reward, terminated, truncated, info)."""
        if self.finished:
            raise RuntimeError("Episode is over. Call reset() first.")
        if action not in range(self.n_actions):
            raise ValueError(f"Action must be an integer from 0 to {self.n_actions - 1}")

        self.steps += 1
        reward = -0.1  # Small cost for taking a step.
        terminated = False
        event = "moved"

        if action in self.MOVES:
            dx, dy = self.MOVES[action]
            x, y = self.player_pos
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.player_pos = (nx, ny)
            else:
                reward -= 1.0  # Walking into a boundary is discouraged.
                event = "boundary"

            if not self.has_ball and self.player_pos == self.ball_pos:
                self.has_ball = True
                reward += 5.0
                event = "ball_collected"

        elif action == self.SHOOT:
            # In Phase 1, a shot scores only from the cell left of the goal.
            shooting_spot = (self.goal_pos[0] - 1, self.goal_pos[1])
            if self.has_ball and self.player_pos == shooting_spot:
                reward += 100.0
                terminated = True
                event = "goal"
            else:
                reward -= 1.0
                event = "missed_shot"

        truncated = self.steps >= self.max_steps and not terminated
        self.finished = terminated or truncated
        if truncated:
            event = "time_limit"

        return self._get_state(), reward, terminated, truncated, {"event": event}

    def render(self):
        """Return a printable text pitch (P player, B ball, G goal)."""
        rows = []
        for y in range(self.height):
            cells = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.player_pos:
                    symbol = "P"
                elif pos == self.ball_pos and not self.has_ball:
                    symbol = "B"
                elif pos == self.goal_pos:
                    symbol = "G"
                else:
                    symbol = "."
                cells.append(symbol)
            rows.append(" ".join(cells))
        possession = "yes" if self.has_ball else "no"
        return "\n".join(rows) + f"\nBall possession: {possession}"


if __name__ == "__main__":
    # A simple test: collect the ball, approach the goal, and shoot.
    env = FootballEnv()
    state, info = env.reset()
    print("Starting state:", state)
    print(env.render())

    actions = [env.RIGHT] * 5 + [env.SHOOT]
    for action in actions:
        state, reward, terminated, truncated, info = env.step(action)
        print(f"\nAction: {env.ACTION_NAMES[action]} | Reward: {reward:.1f} | Event: {info['event']}")
        print(env.render())
        if terminated or truncated:
            print("Episode finished!")
            break
