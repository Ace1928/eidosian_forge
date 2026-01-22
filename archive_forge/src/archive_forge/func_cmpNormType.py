from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def cmpNormType(s, rdvs, lev):
    if rdvs:
        durs = [dur for dur, tmod in s.tupnts if dur > 0]
        ndur = sum(durs) // s.tmnum
        s.irrtup = any((dur != ndur for dur in durs))
        tix = 16 * s.divisions // ndur
        if tix in s.typeMap:
            s.ntype = str(s.typeMap[tix])
        else:
            s.irrtup = 0
    if s.irrtup:
        for dur, tmod in s.tupnts:
            addElemT(tmod, 'normal-type', s.ntype, lev + 1)
    s.tupnts = []