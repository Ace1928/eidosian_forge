from pyparsing import *
import random
import string
def validateItemName(self, s, l, t):
    iname = ' '.join(t)
    if iname not in Item.items:
        raise AppParseException(s, l, "No such item '%s'." % iname)
    return iname