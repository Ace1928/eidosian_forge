from pyparsing import (Regex, Suppress, ZeroOrMore, Group, Optional, Forward,
def bracketed(expr):
    """ Return matcher for `expr` between curly brackets or parentheses """
    return LPAREN + expr + RPAREN | LCURLY + expr + RCURLY