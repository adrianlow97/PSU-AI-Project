#!/usr/bin/env python

__author__ = "Adrian Low, Christopher Renslow, Samuel Shippey"
__version__ = "1.0.0"
__email__ = "crenslow@pdx.edu, lowad@pdx.edu, sshippey@pdx.edu"

import gym

# Global Vars
NUM_EPISODES = 10000
EPS = 0.1 # determine probability of choosing greedy action
GAMMA = 0.9 # for discounted reward q-matrix update
ETA = 0.2 # for q-matrix update


def init_q_matrix(num_states : int, num_actions: int):
    """
        Creates the Q-Matrix for the RL algorithm
        :param num_states: defines number of rows in matrix
        :param num_actions: defines number of columns in matrix
        :return: returns the Q-Matrix full of zero values
        """
    q_mat = []
    for x in range(num_states):
        row = []
        for y in range(num_actions):
            row.append(0)
        q_mat.append(row)
    return q_mat


def determine_action():
    pass


if __name__ == "__main__":

    env = gym.make('Blackjack-v0')


"""
Links to anything that will help our project:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
Video of a guy using this env - https://www.youtube.com/watch?v=e8ofon3sg8E
"""