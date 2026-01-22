from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def fixDoctype(elem):
    if python3:
        xs = E.tostring(elem, encoding='unicode')
    else:
        xs = E.tostring(elem, encoding='utf-8')
    ys = xs.split('\n')
    ys.insert(0, xmlVersion)
    ys.insert(1, '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">')
    return '\n'.join(ys)