from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def delAttrs(part):
    xs = [(m, e) for m in part.findall('measure') for e in m.findall('attributes')]
    for m, e in xs:
        for c in list(e):
            if c.tag == 'clef':
                continue
            if c.tag == 'staff-details':
                continue
            e.remove(c)
        if len(list(e)) == 0:
            m.remove(e)