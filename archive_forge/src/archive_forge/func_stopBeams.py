from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def stopBeams(s):
    if not s.prevNote:
        return
    pbm = s.prevNote.find('beam')
    if pbm != None:
        if pbm.text == 'begin':
            s.prevNote.remove(pbm)
        elif pbm.text == 'continue':
            pbm.text = 'end'
    s.prevNote = None