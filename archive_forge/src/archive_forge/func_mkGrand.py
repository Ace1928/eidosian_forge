from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkGrand(p, vdefs):
    xs = []
    for i, x in enumerate(p.objs):
        if type(x) == pObj:
            us = mkGrand(x, vdefs)
            if x.name == 'grand':
                vids = [y.objs[0] for y in x.objs[1:]]
                nms = [vdefs[u][0] for u in vids if u in vdefs]
                accept = sum([1 for nm in nms if nm]) == 1
                if accept or us[0] == '{*':
                    xs.append(us[1:])
                    vdefs = setFristVoiceNameFromGroup(vids, vdefs)
                    p.objs[i] = x.objs[1]
                else:
                    xs.extend(us[1:])
            else:
                xs.extend(us)
        else:
            xs.append(p.t[0])
    return xs