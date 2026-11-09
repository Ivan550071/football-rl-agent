"""Pygame visualization for the Football RL project (Phase 2.2).

Only draws the world. It NEVER moves the player, computes rewards, or
updates the Q-table. All game rules remain in FootballEnv.
"""

import pygame


class FootballRenderer:
    """Display the existing 7x5 grid as a graphical football pitch."""

    CELL_SIZE = 88
    MARGIN_X = 78
    PITCH_TOP = 112
    BOTTOM_PANEL = 110

    DARK = (17, 31, 39)
    WHITE = (245, 249, 250)
    MUTED = (175, 194, 200)
    BLUE = (55, 145, 242)
    YELLOW = (254, 217, 103)

    def __init__(self, width=7, height=5):
        pygame.init()
        pygame.font.init()

        self.grid_width = width
        self.grid_height = height
        self.pitch_w = width * self.CELL_SIZE
        self.pitch_h = height * self.CELL_SIZE
        self.screen_w = self.pitch_w + 2 * self.MARGIN_X
        self.screen_h = self.PITCH_TOP + self.pitch_h + self.BOTTOM_PANEL

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Football RL Agent | Phase 2")
        self.font = pygame.font.SysFont("consolas", 21)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.title_font = pygame.font.SysFont("consolas", 25, bold=True)

    def _cell_center(self, pos):
        x, y = pos
        return (
            self.MARGIN_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.PITCH_TOP + y * self.CELL_SIZE + self.CELL_SIZE // 2,
        )

    def _label(self, message, xy, color=None, font=None):
        font = font or self.font
        rendered = font.render(str(message), True, color or self.WHITE)
        self.screen.blit(rendered, xy)

    def _pitch(self, goal_pos):
        rect = pygame.Rect(
            self.MARGIN_X, self.PITCH_TOP, self.pitch_w, self.pitch_h
        )

        # Alternating grass strips and subtle grid lines.
        for col in range(self.grid_width):
            color = (31, 126, 74) if col % 2 == 0 else (36, 139, 80)
            pygame.draw.rect(
                self.screen, color,
                (rect.left + col * self.CELL_SIZE, rect.top,
                 self.CELL_SIZE, self.pitch_h),
            )
        pygame.draw.rect(self.screen, self.WHITE, rect, 3)

        cx = rect.centerx
        cy = rect.centery
        pygame.draw.line(self.screen, self.WHITE, (cx, rect.top), (cx, rect.bottom), 2)
        pygame.draw.circle(self.screen, self.WHITE, (cx, cy), self.CELL_SIZE, 2)
        pygame.draw.circle(self.screen, self.WHITE, (cx, cy), 4)

        # Penalty box / goal; goal cell still exists in the original grid.
        penalty = pygame.Rect(
            rect.right - 2 * self.CELL_SIZE,
            rect.centery - self.CELL_SIZE * 1.5,
            2 * self.CELL_SIZE,
            3 * self.CELL_SIZE,
        )
        pygame.draw.rect(self.screen, self.WHITE, penalty, 2)
        goal_y = self._cell_center(goal_pos)[1]
        mouth = pygame.Rect(rect.right, goal_y - 24, 17, 48)
        pygame.draw.rect(self.screen, (210, 233, 237), mouth, 3)
        pygame.draw.circle(
            self.screen, self.YELLOW, self._cell_center(goal_pos), 14, 2
        )

        grid_color = (86, 161, 121)
        for col in range(1, self.grid_width):
            x = rect.left + col * self.CELL_SIZE
            pygame.draw.line(self.screen, grid_color, (x, rect.top), (x, rect.bottom), 1)
        for row in range(1, self.grid_height):
            y = rect.top + row * self.CELL_SIZE
            pygame.draw.line(self.screen, grid_color, (rect.left, y), (rect.right, y), 1)

    def draw(self, game, mode="manual", total_reward=0.0,
             event="ready", finished=False):
        """Render one frame using the state of a FootballEnv instance."""
        self.screen.fill(self.DARK)
        self._label("FOOTBALL RL AGENT", (self.MARGIN_X, 20), font=self.title_font)
        self._label(
            f"MODE: {mode.upper()}    STEPS: {game.steps}/{game.max_steps}",
            (self.MARGIN_X, 66), self.MUTED, self.small_font,
        )

        self._pitch(game.goal_pos)
        center = self._cell_center(game.player_pos)
        pygame.draw.circle(self.screen, (16, 52, 77), (center[0] + 3, center[1] + 4), 20)
        pygame.draw.circle(self.screen, self.BLUE, center, 19)
        pygame.draw.circle(self.screen, self.WHITE, center, 19, 2)
        player_label = self.font.render("P", True, self.WHITE)
        self.screen.blit(player_label, player_label.get_rect(center=center))

        if game.has_ball:
            ball_center = (center[0] + 18, center[1] + 18)
        else:
            ball_center = self._cell_center(game.ball_pos)
        pygame.draw.circle(self.screen, (19, 28, 32), ball_center, 10)
        pygame.draw.circle(self.screen, self.WHITE, ball_center, 8)
        pygame.draw.circle(self.screen, (35, 46, 52), ball_center, 8, 1)

        bottom_y = self.PITCH_TOP + self.pitch_h + 17
        self._label(
            f"BALL: {'YES' if game.has_ball else 'NO '}  "
            f"  REWARD: {total_reward:+.1f}  EVENT: {event}",
            (self.MARGIN_X, bottom_y), self.WHITE, self.small_font,
        )
        if finished:
            message = "GOAL! Press R to play again" if event == "goal" else "TIME UP! Press R to restart"
            self._label(message, (self.MARGIN_X, bottom_y + 28), self.YELLOW, self.small_font)
        else:
            self._label(
                "Arrows/WASD: move | SPACE: shoot | R: reset | TAB: switch mode | ESC: quit",
                (24, bottom_y + 35), self.MUTED, self.small_font,
            )

        pygame.display.flip()

    def close(self):
        pygame.quit()
