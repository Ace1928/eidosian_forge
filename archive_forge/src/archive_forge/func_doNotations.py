from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doNotations(s, n, decos, ptup, alter, tupnotation, tstop, nt, lev):
    slurs = getattr(n, 'slurs', 0)
    pts = getattr(n, 'pitches', [])
    ov = s.overlayVnum
    if pts:
        if type(pts.pitch) == pObj:
            pts = [pts.pitch]
        else:
            pts = [tuple(p.t[-2:]) for p in pts.pitch]
    for pt, (tie_alter, nts, vnum, ntelm) in sorted(list(s.ties.items())):
        if vnum != s.overlayVnum:
            continue
        if pts and pt in pts:
            continue
        if getattr(n, 'chord', 0):
            continue
        if pt == ptup:
            continue
        if getattr(n, 'grace', 0):
            continue
        info('tie between different pitches: %s%s converted to slur' % pt)
        del s.ties[pt]
        e = [t for t in ntelm.findall('tie') if t.get('type') == 'start'][0]
        ntelm.remove(e)
        e = [t for t in nts.findall('tied') if t.get('type') == 'start'][0]
        e.tag = 'slur'
        slurnum = pushSlur(s.slurstack, ov)
        e.set('number', str(slurnum))
        if slurs:
            slurs.t.append(')')
        else:
            slurs = pObj('slurs', [')'])
    tstart = getattr(n, 'tie', 0)
    if not (tstop or tstart or decos or slurs or s.slurbeg or tupnotation or s.trem):
        return nt
    nots = E.Element('notations')
    if s.trem:
        if s.trem < 0:
            tupnotation = 'single'
            s.trem = -s.trem
        if not tupnotation:
            return
        orn = E.Element('ornaments')
        trm = E.Element('tremolo', type=tupnotation)
        trm.text = str(s.trem)
        addElem(nots, orn, lev + 1)
        addElem(orn, trm, lev + 2)
        if tupnotation == 'stop' or tupnotation == 'single':
            s.trem = 0
    elif tupnotation:
        tup = E.Element('tuplet', type=tupnotation)
        if tupnotation == 'start':
            tup.set('bracket', 'yes')
        addElem(nots, tup, lev + 1)
    if tstop:
        del s.ties[ptup]
        tie = E.Element('tied', type='stop')
        addElem(nots, tie, lev + 1)
    if tstart:
        s.ties[ptup] = (alter, nots, s.overlayVnum, nt)
        tie = E.Element('tied', type='start')
        if tstart.t[0] == '.-':
            tie.set('line-type', 'dotted')
        addElem(nots, tie, lev + 1)
    if decos:
        slurMap = {'(': 1, '.(': 1, '(,': 1, "('": 1, '.(,': 1, ".('": 1}
        arts = []
        for d in decos:
            if d in slurMap:
                s.slurbeg.append(d)
                continue
            elif d == 'fermata' or d == 'H':
                ntn = E.Element('fermata', type='upright')
            elif d == 'arpeggio':
                ntn = E.Element('arpeggiate', number='1')
            elif d in ['~(', '~)']:
                if d[1] == '(':
                    tp = 'start'
                    s.glisnum += 1
                    gn = s.glisnum
                else:
                    tp = 'stop'
                    gn = s.glisnum
                    s.glisnum -= 1
                if s.glisnum < 0:
                    s.glisnum = 0
                    continue
                ntn = E.Element('glissando', {'line-type': 'wavy', 'number': '%d' % gn, 'type': tp})
            elif d in ['-(', '-)']:
                if d[1] == '(':
                    tp = 'start'
                    s.slidenum += 1
                    gn = s.slidenum
                else:
                    tp = 'stop'
                    gn = s.slidenum
                    s.slidenum -= 1
                if s.slidenum < 0:
                    s.slidenum = 0
                    continue
                ntn = E.Element('slide', {'line-type': 'solid', 'number': '%d' % gn, 'type': tp})
            else:
                arts.append(d)
                continue
            addElem(nots, ntn, lev + 1)
        if arts:
            rest = s.doArticulations(nt, nots, arts, lev + 1)
            if rest:
                info('unhandled note decorations: %s' % rest)
    if slurs:
        for d in slurs.t:
            if not s.slurstack.get(ov, 0):
                break
            slurnum = s.slurstack[ov].pop()
            slur = E.Element('slur', number='%d' % slurnum, type='stop')
            addElem(nots, slur, lev + 1)
    while s.slurbeg:
        stp = s.slurbeg.pop(0)
        slurnum = pushSlur(s.slurstack, ov)
        ntn = E.Element('slur', number='%d' % slurnum, type='start')
        if '.' in stp:
            ntn.set('line-type', 'dotted')
        if ',' in stp:
            ntn.set('placement', 'below')
        if "'" in stp:
            ntn.set('placement', 'above')
        addElem(nots, ntn, lev + 1)
    if list(nots) != []:
        addElem(nt, nots, lev)