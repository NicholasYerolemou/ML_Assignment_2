
import re
from sqlalchemy import column
from torch import initial_seed
import FourRooms
from random import randrange
import numpy as np
import random
# https://www.analyticsvidhya.com/blog/2021/04/q-learning-algorithm-with-step-by-step-implementation-using-python/


# epoch ends when you reach terminal state


# find the action with the highest reward, returns a num 0-3 representing an action


FR = FourRooms.FourRooms('simple', False)

rows = 13
columns = 13
# (state,(Q value for all 4 actions))(state size, action size)
Q_values = np.zeros((rows*columns, 4))
actions = [FR.UP, FR.DOWN, FR.LEFT, FR.RIGHT]

ac = ["up", 'down', 'left', 'right']
# initialises the rewards for every block to -1
rewards = np.full((rows, columns), -1)


epsilon = 0.2  # makes a random choice 20% of the time

# Q value = adding the maximum reward attainable from future states to the reward for achieving its current state


def convert(pos):
    return pos[0] + 13*pos[1]


def chooseAction(FR):
    # print("Choose functionQ values for state", FR.getPosition(),
    # ":", Q_values[convert(FR.getPosition())])
    if random.uniform(0, 1) < epsilon:  # random action
        return random.randint(0, 3)

    else:  # highest Q action
        return np.argmax(Q_values[convert(FR.getPosition())])


# initialise Q values
discount_factor = 0.8
learning_rate = 0.8

episodes = 100
numMoves = 0
counter = 0

initialState = FR.getPosition()
print("Agent initial state", initialState)
for episodes in range(episodes):
    FR.newEpoch()

    print("episode", episodes)
    counter = 0
    while(FR.isTerminal() == False):
       # print("QQQQQQ", Q_values[FR.getPosition(), :])
        numMoves += 1
        counter += 1

        # choose an action

        actionNum = chooseAction(FR)
        # print("we are in state", FR.getPosition())
        # print("we are moving:", ac[actionNum])
        oldState = FR.getPosition()  # the old state before our action
        cellType, state, packagesLeft, isTerminal = FR.takeAction(
            actions[actionNum])  # if the action takes me into a non traversable space the same position is returned

        if(cellType == 1 or cellType == 2 or cellType == 3):  # the cell type is a package
            reward = 100
        elif(oldState == state):
            reward = -100
        else:
            reward = -1  # rewards[state]

        # print("moved to state", state, "with reward", reward)

        # update Q values

        Q_values[convert(oldState), actionNum] = Q_values[convert(oldState), actionNum] + learning_rate * \
            (reward + discount_factor *
             np.max(Q_values[convert(state), :])-Q_values[convert(oldState), actionNum])
    print(numMoves)
    numMoves = 0


# print(Q_values)
FR.showPath(-1)
