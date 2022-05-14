import FourRooms


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
