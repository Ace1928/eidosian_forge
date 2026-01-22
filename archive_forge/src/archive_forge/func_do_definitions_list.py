from pyparsing import *
def do_definitions_list(str, loc, toks):
    toks = toks.asList()
    if len(toks) > 1:
        return Or(toks)
    else:
        return [toks[0]]