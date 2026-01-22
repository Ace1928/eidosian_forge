import pyparsing as pp
from operator import mul
from functools import reduce
def makeLit(s, val):
    ret = pp.CaselessLiteral(s)
    return ret.setParseAction(pp.replaceWith(val))