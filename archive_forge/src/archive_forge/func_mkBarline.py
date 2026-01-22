from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkBarline(s, maat, loc, lev, style='', dir='', ending=''):
    b = E.Element('barline', location=loc)
    if style:
        addElemT(b, 'bar-style', style, lev + 1)
    if s.curVolta:
        end = E.Element('ending', number=s.curVolta, type='stop')
        s.curVolta = ''
        if loc == 'left':
            bp = E.Element('barline', location='right')
            addElem(bp, end, lev + 1)
            addElem(s.prevmsre, bp, lev)
        else:
            addElem(b, end, lev + 1)
    if ending:
        ending = ending.replace('-', ',')
        endtxt = ''
        if ending.startswith('"'):
            endtxt = ending.strip('"')
            ending = '33'
        end = E.Element('ending', number=ending, type='start')
        if endtxt:
            end.text = endtxt
        addElem(b, end, lev + 1)
        s.curVolta = ending
    if dir:
        r = E.Element('repeat', direction=dir)
        addElem(b, r, lev + 1)
    addElem(maat, b, lev)