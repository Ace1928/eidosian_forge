from pyparsing import *
def do_repeated_sequence(str, loc, toks):
    return ZeroOrMore(toks[0])