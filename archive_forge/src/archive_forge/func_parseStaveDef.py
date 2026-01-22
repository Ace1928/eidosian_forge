from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def parseStaveDef(s, vdefs):
    for vid in vdefs:
        s.vcepid[vid] = vid
    if not s.staveDefs:
        return vdefs
    for x in s.staveDefs[1:]:
        info('%%%%%s dropped, multiple stave mappings not supported' % x)
    x = s.staveDefs[0]
    score = abc_scoredef.parseString(x)[0]
    f = lambda x: type(x) == uni_type and [x] or x
    s.staves = lmap(f, mkStaves(score, vdefs))
    s.grands = lmap(f, mkGrand(score, vdefs))
    s.groups = mkGroups(score)
    vce_groups = [vids for vids in s.staves if len(vids) > 1]
    d = {}
    for vgr in vce_groups:
        d[vgr[0]] = vgr
    for gstaff in s.grands:
        if len(gstaff) == 1:
            continue
        for v, stf_num in zip(gstaff, range(1, len(gstaff) + 1)):
            for vx in d.get(v, [v]):
                s.gStaffNums[vx] = stf_num
                s.gNstaves[vx] = len(gstaff)
    s.gStaffNumsOrg = s.gStaffNums.copy()
    for xmlpart in s.grands:
        pid = xmlpart[0]
        vces = [v for stf in xmlpart for v in d.get(stf, [stf])]
        for v in vces:
            s.vcepid[v] = pid
    return vdefs