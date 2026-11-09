import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.football_env import FootballEnv


class FootballGymEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi", "human"],
        "render_fps": 4
    }

    def __init__(self, max_steps=60, random_start=True,
                 render_mode=None):
        super().__init__()

        if render_mode not in (
            None, *self.metadata["render_modes"]
        ):
            raise ValueError(
                f"Unsupported render_mode: {render_mode}"
            )

        # Reuse our existing football environment
        self.game = FootballEnv(max_steps=max_steps)
        self.random_start = random_start
        self.render_mode = render_mode

        # Five actions: up, down, left, right, shoot
        self.action_space = spaces.Discrete(
            self.game.n_actions
        )

        # Observation: [player_x, player_y, has_ball]
        self.observation_space = spaces.MultiDiscrete(
            [self.game.width, self.game.height, 2]
        )

        self.start_positions = [
            (x, y)
            for x in range(self.game.width)
            for y in range(self.game.height)
            if (x, y) not in (
                self.game.ball_pos,
                self.game.goal_pos
            )
        ]

    def _get_obs(self):
        x, y = self.game.player_pos
        return np.array(
            [x, y, int(self.game.has_ball)],
            dtype=np.int64
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()

        if self.random_start:
            index = int(
                self.np_random.integers(
                    len(self.start_positions)
                )
            )
            self.game.player_pos = self.start_positions[index]

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = (
            self.game.step(int(action))
        )

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info
        )

    def render(self):
        pitch = self.game.render()

        if self.render_mode == "human":
            print(pitch)
            return None

        return pitch

    def close(self):
        pass


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = FootballGymEnv(random_start=True)
    check_env(env)

    print("Gymnasium environment check passed!")

    obs, info = env.reset(seed=42)

    print("Starting observation:", obs)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print(env.render())

    env.close()
