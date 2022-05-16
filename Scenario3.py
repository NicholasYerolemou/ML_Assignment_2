import sys
import re
from sqlalchemy import column
from torch import initial_seed
import FourRooms
from random import randrange
import numpy as np
import random


stock = False
if(len(sys.argv) > 1):
    if(sys.argv[1] == "-stochastic"):
        stock = True
    else:
        stock = False

FR = FourRooms.FourRooms('rgb', stock)

rows = 13
columns = 13


# ***********************************

# Q values to find each package
# when search for red package other two are avoid with heavy negative rewards
red_Q_values = np.zeros((rows*columns, 4))
green_Q_values = np.zeros((rows*columns, 4))
blue_Q_values = np.zeros((rows*columns, 4))

Q = [blue_Q_values, green_Q_values, red_Q_values]

# **********************************


actions = [FR.UP, FR.DOWN, FR.LEFT, FR.RIGHT]

epsilon = 0.4  # makes a random choice 20% of the time


def convert(pos):  # converts coord to array position
    return pos[0] + 13*pos[1]


def chooseAction(FR, Q):

    if random.uniform(0, 1) < epsilon:  # random action
        return random.randint(0, 3)

    else:  # highest Q action
        return np.argmax(Q[convert(FR.getPosition())])


# initialise Q values
discount_factor = 0.8
learning_rate = 0.8

episodes = 1000
numMoves = 0

packagesLeft = 3
initialState = FR.getPosition()


print("Agent initial state", initialState)
for episodes in range(episodes):
    order = []  # order the packages were collected in
    FR.newEpoch()

    print("episode", episodes)
    counter = 0
    while(FR.isTerminal() == False):
        numMoves += 1
        counter += 1

        # choose an action

        # the Q of the packet we are looking for is given
        actionNum = chooseAction(FR, Q[packagesLeft-1])
        oldState = FR.getPosition()  # the old state before our action
        cellType, state, packagesLeft, isTerminal = FR.takeAction(
            actions[actionNum])  # if the action takes me into a non traversable space the same position is returned

        if(cellType == 1 or cellType == 2 or cellType == 3):  # the cell type is a package
            if(cellType == 1):
                order.append("red")
                #print("red found")
               # print("the green Q values are", Q[packagesLeft-1])
            elif(cellType == 2):
                order.append("green")
                #print("green found")
                #print("the blue Q values are", Q[packagesLeft-1])
            elif(cellType == 3):
                order.append("blue")
                #print("blue found")

            # rewards depend on which package we are searching for
            if(packagesLeft == 2):  # foumd the red package
                reward = 100
            else:
                reward = -100  # avoid this package

            if(packagesLeft == 1):  # foumd the green package
                reward = 100
            else:
                reward = -100  # avoid this package

            if(packagesLeft == 0):  # found the blue package
                reward = 100
            else:
                reward = -100  # avoid this package

        elif(oldState == state):
            reward = -100
        else:
            reward = -1  # rewards[state]

        # update Q values

        Q[packagesLeft-1][convert(oldState), actionNum] = Q[packagesLeft-1][convert(oldState), actionNum] + learning_rate * \
            (reward + discount_factor *
             np.max(Q[packagesLeft-1][convert(state), :])-Q[packagesLeft-1][convert(oldState), actionNum])
    print("num moves", numMoves)
    print(order)
    numMoves = 0


# print(Q_values)
FR.showPath(-1)
