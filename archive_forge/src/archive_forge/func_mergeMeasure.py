from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mergeMeasure(m1, m2, slur_offset, voice_offset, rOpt, is_grand=0, is_overlay=0):
    slurs = m2.findall('note/notations/slur')
    for slr in slurs:
        slrnum = int(slr.get('number')) + slur_offset
        slr.set('number', str(slrnum))
    vs = m2.findall('note/voice')
    for v in vs:
        v.text = str(voice_offset + int(v.text))
    ls = m1.findall('note/lyric')
    lnum_max = max([int(l.get('number')) for l in ls] + [0])
    ls = m2.findall('note/lyric')
    for el in ls:
        n = int(el.get('number'))
        el.set('number', str(n + lnum_max))
    ns = m1.findall('note')
    dur1 = sum((int(n.find('duration').text) for n in ns if n.find('grace') == None and n.find('chord') == None))
    dur1 -= sum((int(b.text) for b in m1.findall('backup/duration')))
    repbar, nns, es = (0, 0, [])
    for e in list(m2):
        if e.tag == 'attributes':
            if not is_grand:
                continue
            else:
                nns += 1
        if e.tag == 'print':
            continue
        if e.tag == 'note' and (rOpt or e.find('rest') == None):
            nns += 1
        if e.tag == 'barline' and e.find('repeat') != None:
            repbar = e
        es.append(e)
    if nns > 0:
        if dur1 > 0:
            b = E.Element('backup')
            addElem(m1, b, level=3)
            addElemT(b, 'duration', str(dur1), level=4)
        for e in es:
            addElem(m1, e, level=3)
    elif is_overlay and repbar:
        addElem(m1, repbar, level=3)