from sqlalchemy import column
import FourRooms
from random import randrange
import numpy as np
# https://www.analyticsvidhya.com/blog/2021/04/q-learning-algorithm-with-step-by-step-implementation-using-python/
"""
FR = FourRooms('simple', bool=False)


pos = (int, int)
pos = FR.getPosition()  # gets position


packagesLeft = FR.getPackagesRemaining()


cellType, newPos, packagesLeft, isTerminal = FR.takeAction(FR.UP)


isTerminal = FR.isTerminal()  # True if sim is over

FR.newEpoch()  # resets the enviroment to starting positions


# index’th epoch.
# showPath(-1) will show the agent’s path for the last recorded epoch.
# displays the agent path using the matplotlib GUI

epoch = 0
FR.showPath(epoch, str='path')
"""
"""
# Action Constants
FourRooms.UP = 0
FourRooms.DOWN = 1
FourRooms.LEFT = 2
FourRooms.RIGHT = 3

# Grid Cell Type Constants
FourRooms.EMPTY = 0
FourRooms.RED = 1
FourRooms.GREEN = 2
FourRooms.BLUE = 3
"""


# epoch ends when you reach terminal state


gamma = 0.8
reward = 0
action = 0
state = (int, int)


def chooseAction(FR):
    num = randrange(4)
    num = num - 1
    if(num == 0):
        return FR.UP
    if(num == 1):
        return FR.DOWN
    if(num == 2):
        return FR.LEFT
    if(num == 3):
        return FR.RIGHT


FR = FourRooms.FourRooms('simple', False)

rows = 13
columns = 13

Q_values = np.zeros((rows, columns, 4))  # (state,(Q value for all 4 actions))
actions = [FR.UP, FR.DOWN, FR.LEFT, FR.RIGHT]

# initialises the rewards for every block to -1
rewards = np.full((rows, columns), -1)
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

FR.newEpoch
epoch = 1
print("Epoch", epoch)
print("Packages left", FR.getPackagesRemaining())
print("Agent initial position", FR.getPosition())
counter = 0
while(FR.isTerminal() == False and counter < 10):
    cellType, newPos, packagesLeft, isTerminal = FR.takeAction(
        chooseAction(FR))
    print(FR.getPosition())
    print(cellType)
    counter += 1
FR.showPath(0)
