import re
from . import constants as const
def readMPSSetRhs(line, constraintsDict):
    constraintsDict[line[1]]['constant'] = -float(line[2])
    if len(line) == 5:
        constraintsDict[line[3]]['constant'] = -float(line[4])
    return