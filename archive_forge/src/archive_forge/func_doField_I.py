from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doField_I(s, type, x, instDir, addTrans):

    def instChange(midchan, midprog):
        if midchan and midchan != s.midprg[0]:
            instDir('midi-channel', midchan, 'chan: %s')
        if midprog and midprog != s.midprg[1]:
            instDir('midi-program', str(int(midprog) + 1), 'prog: %s')

    def readPfmt(x, n):
        if not s.pageFmtAbc:
            s.pageFmtAbc = s.pageFmtDef
        ro = re.search('[^.\\d]*([\\d.]+)\\s*(cm|in|pt)?', x)
        if ro:
            x, unit = ro.groups()
            u = {'cm': 10.0, 'in': 25.4, 'pt': 25.4 / 72}[unit] if unit else 1.0
            s.pageFmtAbc[n] = float(x) * u
        else:
            info('error in page format: %s' % x)

    def readPercMap(x):

        def getMidNum(sndnm):
            pnms = sndnm.split('-')
            ps = s.percsnd[:]
            _f = lambda ip, xs, pnm: ip < len(xs) and xs[ip].find(pnm) > -1
            for ip, pnm in enumerate(pnms):
                ps = [(nm, mnum) for nm, mnum in ps if _f(ip, nm.split('-'), pnm)]
                if len(ps) <= 1:
                    break
            if len(ps) == 0:
                info('drum sound: %s not found' % sndnm)
                return '38'
            return ps[0][1]

        def midiVal(acc, step, oct):
            oct = (4 if step.upper() == step else 5) + int(oct)
            return oct * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step.upper())] + {'^': 1, '_': -1, '=': 0}.get(acc, 0) + 12
        p0, p1, p2, p3, p4 = abc_percmap.parseString(x).asList()
        acc, astep, aoct = p1
        nstep, noct = (astep, aoct) if p2 == '*' else p2
        if p3 == '*':
            midi = str(midiVal(acc, astep, aoct))
        elif isinstance(p3, list_type):
            midi = str(midiVal(p3[0], p3[1], p3[2]))
        elif isinstance(p3, int_type):
            midi = str(p3)
        else:
            midi = getMidNum(p3.lower())
        head = re.sub('(.)-([^x])', '\\1 \\2', p4)
        s.percMap[s.pid, acc + astep, aoct] = (nstep, noct, midi, head)
    if x.startswith('score') or x.startswith('staves'):
        s.staveDefs += [x]
    elif x.startswith('staffwidth'):
        info('skipped I-field: %s' % x)
    elif x.startswith('staff'):
        r1 = re.search('staff *([+-]?)(\\d)', x)
        if r1:
            sign = r1.group(1)
            num = int(r1.group(2))
            gstaff = s.gStaffNums.get(s.vid, 0)
            if sign:
                num = sign == '-' and gstaff - num or gstaff + num
            else:
                try:
                    vabc = s.staves[num - 1][0]
                except:
                    vabc = 0
                    info('abc staff %s does not exist' % num)
                num = s.gStaffNumsOrg.get(vabc, 0)
            if gstaff and num > 0 and (num <= s.gNstaves[s.vid]):
                s.gStaffNums[s.vid] = num
            else:
                info('could not relocate to staff: %s' % r1.group())
        else:
            info('not a valid staff redirection: %s' % x)
    elif x.startswith('scale'):
        readPfmt(x, 0)
    elif x.startswith('pageheight'):
        readPfmt(x, 1)
    elif x.startswith('pagewidth'):
        readPfmt(x, 2)
    elif x.startswith('leftmargin'):
        readPfmt(x, 3)
    elif x.startswith('rightmargin'):
        readPfmt(x, 4)
    elif x.startswith('topmargin'):
        readPfmt(x, 5)
    elif x.startswith('botmargin'):
        readPfmt(x, 6)
    elif x.startswith('MIDI') or x.startswith('midi'):
        r1 = re.search('program *(\\d*) +(\\d+)', x)
        r2 = re.search('channel *(\\d+)', x)
        r3 = re.search("drummap\\s+([_=^]*)([A-Ga-g])([,']*)\\s+(\\d+)", x)
        r4 = re.search('control *(\\d+) +(\\d+)', x)
        ch_nw, prg_nw, vol_nw, pan_nw = ('', '', '', '')
        if r1:
            ch_nw, prg_nw = r1.groups()
        if r2:
            ch_nw = r2.group(1)
        if r4:
            cnum, cval = r4.groups()
            if cnum == '7':
                vol_nw = cval
            if cnum == '10':
                pan_nw = cval
        if r1 or r2 or r4:
            ch = ch_nw or s.midprg[0]
            prg = prg_nw or s.midprg[1]
            vol = vol_nw or s.midprg[2]
            pan = pan_nw or s.midprg[3]
            instId = 'I%s-%s' % (s.pid, s.vid)
            if instId in s.midiInst:
                instChange(ch, prg)
            s.midprg = [ch, prg, vol, pan]
        if r3:
            acc, step, oct, midi = r3.groups()
            oct = -len(oct) if ',' in x else len(oct)
            notehead = 'x' if acc == '^' else 'circle-x' if acc == '_' else 'normal'
            s.percMap[s.pid, acc + step, oct] = (step, oct, midi, notehead)
        r = re.search('transpose[^-\\d]*(-?\\d+)', x)
        if r:
            addTrans(r.group(1))
    elif x.startswith('percmap'):
        readPercMap(x)
        s.pMapFound = 1
    else:
        info('skipped I-field: %s' % x)