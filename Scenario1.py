import sys
import re
from sqlalchemy import column
from torch import initial_seed
import FourRooms
from random import randrange
import numpy as np
import random
# some inspiration was taken from https://www.analyticsvidhya.com/blog/2021/04/q-learning-algorithm-with-step-by-step-implementation-using-python/

stock = False
if(len(sys.argv) > 1):
    if(sys.argv[1] == "-stochastic"):
        stock = True
    else:
        stock = False


FR = FourRooms.FourRooms('simple', stock)

rows = 13
columns = 13
# (state,(Q value for all 4 actions))(state size, action size)
Q_values = np.zeros((rows*columns, 4))
actions = [FR.UP, FR.DOWN, FR.LEFT, FR.RIGHT]


epsilon = 0.2  # makes a random choice 20% of the time


def convert(pos):  # convert co-ord to array number
    return pos[0] + 13*pos[1]


def chooseAction(FR):
    if random.uniform(0, 1) < epsilon:  # random action
        return random.randint(0, 3)

    else:  # highest Q action
        return np.argmax(Q_values[convert(FR.getPosition())])


# initialise Q values
discount_factor = 0.8
learning_rate = 0.8

episodes = 1000
numMoves = 0
#counter = 0

initialState = FR.getPosition()
print("Agent initial state", initialState)
for episodes in range(episodes):
    FR.newEpoch()

    counter = 0
    while(FR.isTerminal() == False):
        numMoves += 1
        #counter += 1

        # choose an action
        actionNum = chooseAction(FR)
        oldState = FR.getPosition()  # the old state before our action
        cellType, state, packagesLeft, isTerminal = FR.takeAction(
            actions[actionNum])  # if the action takes me into a non traversable space the same position is returned

        if(cellType == 1 or cellType == 2 or cellType == 3):  # the cell type is a package
            reward = 100
        elif(oldState == state):
            reward = -100
        else:
            reward = -1

        # update Q values
        Q_values[convert(oldState), actionNum] = Q_values[convert(oldState), actionNum] + learning_rate * \
            (reward + discount_factor *
             np.max(Q_values[convert(state), :])-Q_values[convert(oldState), actionNum])

    if(episodes % 10 == 0):
        print("episode", episodes)
        print("number of moves taken:", numMoves)
    numMoves = 0

FR.showPath(-1)
