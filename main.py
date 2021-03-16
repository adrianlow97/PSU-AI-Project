#!/usr/bin/env python

__author__ = "Adrian Low, Christopher Renslow, Samuel Shippey"
__version__ = "1.0.0"
__email__ = "crenslow@pdx.edu, lowad@pdx.edu, sshippey@pdx.edu"

import gym
import random
import numpy as np
import matplotlib as plt
import csv

# Global Vars
NUM_EPISODES = 10000
EPS = 0.1  # determine probability of choosing greedy action
GAMMA = 0.9  # for discounted reward q-matrix update
ETA = 0.2  # for q-matrix update


def get_epsilon(curr_episode, factor=100, constant=-1):
    if constant == -1:
        return factor / (factor + curr_episode)
    else:
        return constant


def determine_action(q_table, episode_num, curr_state, sa=0):
    p_greedy = random.random()
    if sa == 0: # if not using simulated annealing for epsilon
        eps = get_epsilon(episode_num, constant=EPS)
    else:
        eps = get_epsilon(episode_num)
    if p_greedy >= eps:
        next_action = q_table[curr_state].index(max(q_table[curr_state]))  # action that maximizes current Q(s,a)
    else:
        next_action = random.randrange(len(q_table[curr_state]))  # random action
    return next_action


def export_q_matrix(q_table, fname):
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        header = ['player hand sum', 'dealers card', 'usable ace', 'stick', 'hit']
        writer.writerow(header)
        for key in q_table.keys():
            row = []
            for x in key:
                row.append(x)
            for y in q_table[key]:
                row.append(y)
            writer.writerow(row)


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    agent_sum_space = [i for i in range(4, 32)]  # possible sums of cards in agents hands 4-21 plus bust hands 22-31
    dealer_card_space = [i for i in range(1, 11)]  # possible values for dealer's card shown 1-10
    usable_ace_space = [0, 1]  # does agent have a usable ace T or F
    agent_action_space = [0, 1]  # agent has only two actions hit or stick

    q = {}

    for agent_sum in agent_sum_space:
        for dealer_card in dealer_card_space:
            for usable_ace in usable_ace_space:
                actions = []
                for agent_action in agent_action_space:
                    actions.append(0)
                q[(agent_sum, dealer_card, usable_ace)] = actions  # initialize everything to 0

    #TRAINING
    total_reward = 0
    for episode in range(NUM_EPISODES):
        state = env.reset()  # deals new hands and returns state (player sum, dealers card, usable ace)
        episode_reward = 0
        done = False
        while done == False:
            action = determine_action(q, episode, state, sa=1)
            new_state, reward, done, desc = env.step(action)
            episode_reward += reward

        # Update Q-Matrix Value
        q[state][action] += ETA * (reward + (GAMMA * max(q[new_state])) - q[state][action])

        total_reward += episode_reward

    print("Average Reward:" + str(total_reward / NUM_EPISODES))
    export_q_matrix(q, "q_matrix.csv")

    #TESTING
    total_reward = 0
    for episode in range(NUM_EPISODES):
        state = env.reset()  # deals new hands and returns state (player sum, dealers card, usable ace)
        episode_reward = 0
        done = False
        while done == False:
            action = determine_action(q, episode, state)
            new_state, reward, done, desc = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    print("Average Reward:" + str(total_reward / NUM_EPISODES))
