from pyparsing import *
import random
import string
def playGame(p, startRoom):
    parser = Parser()
    p.moveTo(startRoom)
    while not p.gameOver:
        cmdstr = input('>> ')
        cmd = parser.parseCmd(cmdstr)
        if cmd is not None:
            cmd.command(p)
    print()
    print('You ended the game with:')
    for i in p.inv:
        print(' -', aOrAn(i))