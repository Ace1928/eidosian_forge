from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def setFristVoiceNameFromGroup(vids, vdefs):
    vids = [v for v in vids if v in vdefs]
    if not vids:
        return vdefs
    vid0 = vids[0]
    _, _, vdef0 = vdefs[vid0]
    for vid in vids:
        nm, snm, vdef = vdefs[vid]
        if nm:
            vdefs[vid0] = (nm, snm, vdef0)
            break
    return vdefs