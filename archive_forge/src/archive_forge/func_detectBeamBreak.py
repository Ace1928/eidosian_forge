from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def detectBeamBreak(line, loc, t):
    global prevloc
    xs = line[prevloc:loc + 1]
    xs = xs.lstrip()
    prevloc = loc
    b = pObj('bbrk', [' ' in xs])
    t.insert(0, b)