from pyparsing import alphanums, nums, quotedString
from pyparsing import Combine, Forward, Group, Literal, oneOf, OneOrMore, Optional, Suppress, ZeroOrMore, Word
from pyparsing import ParseException
def parsePGN(pgn, bnf=pgnGrammar, fn=None):
    try:
        return bnf.parseString(pgn)
    except ParseException as err:
        print(err.line)
        print(' ' * (err.column - 1) + '^')
        print(err)