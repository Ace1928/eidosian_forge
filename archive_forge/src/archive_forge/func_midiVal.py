from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def midiVal(acc, step, oct):
    oct = (4 if step.upper() == step else 5) + int(oct)
    return oct * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step.upper())] + {'^': 1, '_': -1, '=': 0}.get(acc, 0) + 12