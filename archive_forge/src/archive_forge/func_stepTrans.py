from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def stepTrans(step, soct, clef):
    if clef.startswith('bass'):
        nm7 = 'C,D,E,F,G,A,B'.split(',')
        n = 14 + nm7.index(step) - 12
        step, soct = (nm7[n % 7], soct + n // 7 - 2)
    return (step, soct)