import sys
import re
from sqlalchemy import column
from torch import initial_seed
import FourRooms
from random import randrange
import numpy as np
import random
import matplotlib.pyplot as plt


stock = False
if(len(sys.argv) > 1):
    if(sys.argv[1] == "-stochastic"):
        stock = True
    else:
        stock = False

FR = FourRooms.FourRooms('multi', stock)

rows = 13
columns = 13

Q_values = np.zeros((rows*columns, 4))
actions = [FR.UP, FR.DOWN, FR.LEFT, FR.RIGHT]

epsilon = 0.2  # makes a random choice 20% of the time


def convert(pos):  # converts coord to array position
    return pos[0] + 13*pos[1]


def chooseAction(FR):
    if random.uniform(0, 1) < epsilon:  # random action
        return random.randint(0, 3)

    else:  # highest Q action
        return np.argmax(Q_values[convert(FR.getPosition())])


packages_found = []
# initialise Q values
discount_factor = 0.9
learning_rate = 0.9

episodes = 5000
numMoves = 0


minMoves = 100000
# plot = [] #holds the number of actions taken before terminal state for each episode, to be plotted
#x = []


initialState = FR.getPosition()
print("Agent initial state", initialState)
for episodes in range(episodes):
    # x.append(episodes)
    FR.newEpoch()

    counter = 0
    while(FR.isTerminal() == False):
        numMoves += 1

        # choose an action

        actionNum = chooseAction(FR)
        oldState = FR.getPosition()  # the old state before our action
        cellType, state, packagesLeft, isTerminal = FR.takeAction(
            actions[actionNum])  # if the action takes me into a non traversable space the same position is returned

        if(cellType == 1 or cellType == 2 or cellType == 3):  # the cell type is a package
            if(packagesLeft > 0):
                reward = (1/packagesLeft)*100
            else:
                reward = 100
        elif(oldState == state):
            reward = -100
        else:
            reward = -1

        # update Q values
        Q_values[convert(oldState), actionNum] = Q_values[convert(oldState), actionNum] + learning_rate * \
            (reward + discount_factor *
             np.max(Q_values[convert(state), :])-Q_values[convert(oldState), actionNum])
    # plot.append(numMoves)
    if (numMoves < minMoves):
        minMoves = numMoves
    if(episodes % 100 == 0):
        print(print("episode", episodes))
        print("Average number of actions taken for past 100 episodes:", numMoves/100)
        numMoves = 0
        # print()


print("Min number of moves taken over all episodes:", minMoves)
FR.showPath(-1)
#plt.plot(x, plot)
plt.show()
