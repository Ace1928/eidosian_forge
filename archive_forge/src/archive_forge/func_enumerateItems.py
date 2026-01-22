from pyparsing import *
import random
import string
def enumerateItems(l):
    if len(l) == 0:
        return 'nothing'
    out = []
    if len(l) > 1:
        out.append(', '.join((aOrAn(item) for item in l[:-1])))
        out.append('and')
    out.append(aOrAn(l[-1]))
    return ' '.join(out)