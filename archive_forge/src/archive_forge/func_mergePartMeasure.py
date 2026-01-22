from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mergePartMeasure(part, msre, ovrlaynum, rOpt):
    slur_offset = 0
    last_msre = list(part)[-1]
    mergeMeasure(last_msre, msre, slur_offset, ovrlaynum, rOpt, is_overlay=1)