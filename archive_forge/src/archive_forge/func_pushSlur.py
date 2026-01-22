from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def pushSlur(boogStapel, stem):
    if stem not in boogStapel:
        boogStapel[stem] = []
    boognum = sum(map(len, boogStapel.values())) + 1
    boogStapel[stem].append(boognum)
    return boognum