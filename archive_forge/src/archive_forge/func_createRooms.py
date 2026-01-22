from pyparsing import *
import random
import string
def createRooms(rm):
    """
    create rooms, using multiline string showing map layout
    string contains symbols for the following:
     A-Z, a-z indicate rooms, and rooms will be stored in a dictionary by
               reference letter
     -, | symbols indicate connection between rooms
     <, >, ^, . symbols indicate one-way connection between rooms
    """
    ret = {}
    for c in rm:
        if c in string.ascii_letters:
            if c != 'Z':
                ret[c] = Room(c)
            else:
                ret[c] = Exit()
    rows = rm.split('\n')
    for row, line in enumerate(rows):
        for col, c in enumerate(line):
            if c in string.ascii_letters:
                room = ret[c]
                n = None
                s = None
                e = None
                w = None
                if col > 0 and line[col - 1] in '<-':
                    other = line[col - 2]
                    w = ret[other]
                if col < len(line) - 1 and line[col + 1] in '->':
                    other = line[col + 2]
                    e = ret[other]
                if row > 1 and col < len(rows[row - 1]) and (rows[row - 1][col] in '|^'):
                    other = rows[row - 2][col]
                    n = ret[other]
                if row < len(rows) - 1 and col < len(rows[row + 1]) and (rows[row + 1][col] in '|.'):
                    other = rows[row + 2][col]
                    s = ret[other]
                room.doors = [n, s, e, w]
    return ret