import numpy as np
from game.SpaceInvaders import SpaceInvaders
from game.epsilon_profile import EpsilonProfile

class QAgent:
    def __init__(self, game: SpaceInvaders, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        self.game = game
        self.num_actions = game.na  # Les actions possibles : rester immobile, déplacer à gauche, déplacer à droite, tirer
       

        #    Calculer le nombre de lignes et de colonnes en fonction de la taille de l'écran et de la taille d'une cellule
        self.cell_size = 50  # Choisir la taille appropriée pour une cellule
        self.rows = game.screen_height // self.cell_size
        self.cols = game.screen_width // self.cell_size
        
        self.Q = np.zeros([self.rows, self.cols, self.num_actions])        
        
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

    def learn(self, num_episodes, max_steps):
        for episode in range(num_episodes):
            state = self.game.reset()

            for step in range(max_steps):
                action = self.select_action(state)
                _, reward, terminal = self.game.step(action)
                next_state = self.game.get_state()
                self.updateQ(state, action, reward, next_state)

                if terminal:
                    break

                state = next_state

            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (num_episodes - 1.), self.eps_profile.final)


    def updateQ(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        self.Q[state_index[0], state_index[1], action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state_index[0], next_state_index[1], :]) - self.Q[state_index[0], state_index[1], action])


    def select_action(self, state):
        state_index = self.state_to_index(state)

        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.select_greedy_action(state_index)

    def select_greedy_action(self, state_index):
        max_value = np.max(self.Q[state_index])
        return np.random.choice(np.where(self.Q[state_index] == max_value)[0])

    def state_to_index(self, state):
        player_position, invaders_positions, bullet_position, bullet_state = state
        player_x, player_y = player_position
        bullet_x, bullet_y = bullet_position

        player_x_index = min(self.cols - 1, int(player_x // self.cell_size))
        player_y_index = min(self.rows - 1, int(player_y // self.cell_size))
        bullet_x_index = min(self.cols - 1, int(bullet_x // self.cell_size))
        bullet_y_index = min(self.rows - 1, int(bullet_y // self.cell_size))

        return (player_x_index, player_y_index), (bullet_x_index, bullet_y_index)
