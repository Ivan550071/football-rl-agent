"Defining the Q_learning agent with it's hyperparameters."
"The agent uses Bellman update and epsilon-greedy policy"
"Additionally I added Epsilon decay in order to reduce random exploration"
import numpy as np


class QLearningAgent:
    def __init__(self, width, height, n_actions,
                 alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.05):

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

        # Q-table: (x, y, has_ball, action)
        self.q_table = np.zeros(
            (width, height, 2, n_actions)
        )

    def choose_action(self, state, training=True):
        """Choose an action using an epsilon-greedy policy."""

        # Exploration: choose a random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Exploitation: choose an action with the highest Q-value
        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """Update the Q-table using the Q-learning equation."""

        old_q = self.q_table[state][action]

        # No future reward after the episode ends
        if done:
            best_next_q = 0.0
        else:
            best_next_q = np.max(self.q_table[next_state])

        target = reward + self.gamma * best_next_q

        self.q_table[state][action] += self.alpha * (target - old_q)

    def decay_exploration(self):
        """Reduce exploration after each training episode."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
