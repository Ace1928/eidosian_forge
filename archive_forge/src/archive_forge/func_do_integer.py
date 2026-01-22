from pyparsing import *
def do_integer(str, loc, toks):
    return int(toks[0])