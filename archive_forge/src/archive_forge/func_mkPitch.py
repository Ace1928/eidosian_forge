from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkPitch(s, acc, note, oct, lev):
    if s.percVoice:
        octq = int(oct) + s.gtrans
        tup = s.percMap.get((s.pid, acc + note, octq), s.percMap.get(('', acc + note, octq), 0))
        if tup:
            step, soct, midi, notehead = tup
        else:
            step, soct = (note, octq)
        octnum = (4 if step.upper() == step else 5) + int(soct)
        if not tup:
            midi = str(octnum * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step.upper())] + {'^': 1, '_': -1}.get(acc, 0) + 12)
            notehead = {'^': 'x', '_': 'circle-x'}.get(acc, 'normal')
            if s.pMapFound:
                info('no I:percmap for: %s%s in part %s, voice %s' % (acc + note, -oct * ',' if oct < 0 else oct * "'", s.pid, s.vid))
            s.percMap[s.pid, acc + note, octq] = (note, octq, midi, notehead)
        else:
            step, octnum = stepTrans(step.upper(), octnum, s.curClef)
        pitch = E.Element('unpitched')
        addElemT(pitch, 'display-step', step.upper(), lev + 1)
        addElemT(pitch, 'display-octave', str(octnum), lev + 1)
        return (pitch, '', midi, notehead)
    nUp = note.upper()
    octnum = (4 if nUp == note else 5) + int(oct) + s.gtrans
    pitch = E.Element('pitch')
    addElemT(pitch, 'step', nUp, lev + 1)
    alter = ''
    if (note, oct) in s.ties:
        tied_alter, _, vnum, _ = s.ties[note, oct]
        if vnum == s.overlayVnum:
            alter = tied_alter
    elif acc:
        s.msreAlts[nUp, octnum] = s.alterTab[acc]
        alter = s.alterTab[acc]
    elif (nUp, octnum) in s.msreAlts:
        alter = s.msreAlts[nUp, octnum]
    elif nUp in s.keyAlts:
        alter = s.keyAlts[nUp]
    if alter:
        addElemT(pitch, 'alter', alter, lev + 1)
    addElemT(pitch, 'octave', str(octnum), lev + 1)
    return (pitch, alter, '', '')