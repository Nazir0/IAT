from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from game.epsilon_profile import EpsilonProfile


def main():

    n_episodes = 500
    max_steps = 50
    gamma = 1.
    alpha = 0.2
    eps_profile = EpsilonProfile(1.0, 0.1)

    game = SpaceInvaders(display=True)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    controller = QAgent(game, eps_profile, gamma, alpha)
    controller.learn(n_episodes, max_steps)

    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__':
    main()
