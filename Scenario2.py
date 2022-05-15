
import re
from sqlalchemy import column
from torch import initial_seed
import FourRooms
from random import randrange
import numpy as np
import random
# https://www.analyticsvidhya.com/blog/2021/04/q-learning-algorithm-with-step-by-step-implementation-using-python/


# epoch ends when you reach terminal state


def setRewards():
    # sets rewards of walls to -100
    for x in range(13):
        for y in range(13):
            if(y == 0 or y == 12 or x == 0 or x == 12):
                rewards[x, y] = -100
            if(y < 7 and x == 6):
                rewards[x, y] = -100
            if(y > 5 and x == 7):
                rewards[x, y] = -100
            if(x < 12 and y == 6):
                rewards[x, y] = -100

    rewards[3, 6] = -1
    rewards[6, 2] = -1
    rewards[10, 6] = -1
    rewards[7, 9] = -1
    print(rewards)


def chooseAction(FR):  # find the action with the highest reward, returns a num 0-3 representing an action
    if random.uniform(0, 1) < epsilon:  # random action
        return random.randint(0, 3)
    else:  # highest Q action
        return np.argmax(Q_values[FR.getPosition()])


FR = FourRooms.FourRooms('simple', False)


FR.newEpoch()
print(FR.getPosition())

FR.newEpoch()
print(FR.getPosition())


FR.newEpoch()
print(FR.getPosition())


rows = 13
columns = 13
# (state,(Q value for all 4 actions))(state size, action size)
Q_values = np.zeros((rows, columns, 4))
actions = [FR.UP, FR.DOWN, FR.LEFT, FR.RIGHT]

# initialises the rewards for every block to -1
rewards = np.full((rows, columns), -1)
setRewards()  # sets the reward values


epsilon = 0.2  # makes a random choice 20% of the time

# Q value = adding the maximum reward attainable from future states to the reward for achieving its current state


# initialise Q values
discount_factor = 0.9
learning_rate = 0.9

episodes = 100

numMoves = 0

counter = 0

for episodes in range(100):
    FR.newEpoch()
    initialState = FR.getPosition()
    print("Agent initial state", initialState)
    print("episode", episodes)
    counter = 0
    while(FR.isTerminal() == False and counter < 1000000):
        numMoves += 1
        counter += 1

        # choose an action
        actionNum = chooseAction(FR)
        oldState = FR.getPosition()  # the old state before our action
        cellType, state, packagesLeft, isTerminal = FR.takeAction(
            actions[actionNum])
        if(cellType == 1 or cellType == 2 or cellType == 3):  # the cell type is a package
            reward = 100
        else:
            reward = rewards[state]

        reward = rewards[state]  # the reward for the new state

        # update Q values
        Q_values[oldState, actions[actionNum]] = Q_values[oldState, actions[actionNum]] + learning_rate * \
            (reward + discount_factor *
                np.max(Q_values[state, :])-Q_values[oldState, actions[actionNum]])
    print("number of moves:", numMoves)
    numMoves = 0

# print(Q_values)
FR.showPath(-1)
