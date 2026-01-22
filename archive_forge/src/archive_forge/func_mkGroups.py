from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkGroups(p):
    xs = []
    for x in p.objs:
        if type(x) == pObj:
            if x.name == 'vid':
                xs.extend(mkGroups(x))
            elif x.name == 'bracketgr':
                xs.extend(['['] + mkGroups(x) + [']'])
            elif x.name == 'bracegr':
                xs.extend(['{'] + mkGroups(x) + ['}'])
            else:
                xs.extend(mkGroups(x) + ['}'])
        else:
            xs.append(p.t[0])
    return xs