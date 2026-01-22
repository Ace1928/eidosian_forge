from pyparsing import *
import random
import string
def putItemInRoom(i, r):
    if isinstance(r, str):
        r = rooms[r]
    r.addItem(Item.items[i])