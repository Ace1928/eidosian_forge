from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def reduceMids(parts, vidsnew, midiInst):
    for pid, part in zip(vidsnew, parts):
        mids, repls, has_perc = ({}, {}, 0)
        for ipid, ivid, ch, prg, vol, pan in sorted(list(midiInst.values())):
            if ipid != pid:
                continue
            if ch == '10':
                has_perc = 1
                continue
            instId, inst = ('I%s-%s' % (ipid, ivid), (ch, prg))
            if inst in mids:
                repls[instId] = mids[inst]
                del midiInst[instId]
            else:
                mids[inst] = instId
        if len(mids) < 2 and (not has_perc):
            removeElems(part, 'measure/note', 'instrument')
        else:
            for e in part.findall('measure/note/instrument'):
                id = e.get('id')
                if id in repls:
                    e.set('id', repls[id])