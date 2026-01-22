from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def splitHeaderVoices(abctext):
    escField = lambda x: '[' + x.replace(']', '%5d') + ']'
    r1 = re.compile('%.*$')
    r2 = re.compile('^([A-Zw]:.*$)|\\[[A-Zw]:[^]]*]$')
    r3 = re.compile('^%%(?=[^%])')
    xs, nx, mcont, fcont = ([], 0, 0, 0)
    mln = fln = ''
    for x in abctext.splitlines():
        x = x.strip()
        if not x and nx == 1:
            break
        if x.startswith('X:'):
            if nx == 1:
                break
            nx = 1
        x = r3.sub('I:', x)
        x2 = r1.sub('', x)
        while x2.endswith('*') and (not (x2.startswith('w:') or x2.startswith('+:') or 'percmap' in x2)):
            x2 = x2[:-1]
        if not x2:
            continue
        if x2[:2] == 'W:':
            field = x2[2:].strip()
            ftype = mxm.metaMap.get('W', 'W')
            c = mxm.metadata.get(ftype, '')
            mxm.metadata[ftype] = c + '\n' + field if c else field
            continue
        if x2[:2] == '+:':
            fln += x2[2:]
            continue
        ro = r2.match(x2)
        if ro:
            if fcont:
                fcont = x2[-1] == '\\'
                fln += re.sub('^.:(.*?)\\\\*$', '\\1', x2)
                continue
            if fln:
                mln += escField(fln)
            if x2.startswith('['):
                x2 = x2.strip('[]')
            fcont = x2[-1] == '\\'
            fln = x2.rstrip('\\')
            continue
        if nx == 1:
            fcont = 0
            if fln:
                mln += escField(fln)
                fln = ''
            if mcont:
                mcont = x2[-1] == '\\'
                mln += x2.rstrip('\\')
            else:
                if mln:
                    xs.append(mln)
                    mln = ''
                mcont = x2[-1] == '\\'
                mln = x2.rstrip('\\')
            if not mcont:
                xs.append(mln)
                mln = ''
    if fln:
        mln += escField(fln)
    if mln:
        xs.append(mln)
    hs = re.split('(\\[K:[^]]*\\])', xs[0])
    if len(hs) == 1:
        header = hs[0]
        xs[0] = ''
    else:
        header = hs[0] + hs[1]
        xs[0] = ''.join(hs[2:])
    abctext = '\n'.join(xs)
    hfs, vfs = ([], [])
    for x in header[1:-1].split(']['):
        if x[0] == 'V':
            vfs.append(x)
        elif x[:6] == 'I:MIDI':
            vfs.append(x)
        elif x[:9] == 'I:percmap':
            vfs.append(x)
        else:
            hfs.append(x)
    header = '[' + ']['.join(hfs) + ']'
    abctext = ('[' + ']['.join(vfs) + ']' if vfs else '') + abctext
    xs = abctext.split('[V:')
    if len(xs) == 1:
        abctext = '[V:1]' + abctext
    elif re.sub('\\[[A-Z]:[^]]*\\]', '', xs[0]).strip():
        abctext = '[V:1]' + abctext
    r1 = re.compile('\\[V:\\s*(\\S*)[ \\]]')
    vmap = {}
    vorder = {}
    xs = re.split('(\\[V:[^]]*\\])', abctext)
    if len(xs) == 1:
        raise ValueError('bugs ...')
    else:
        pm = re.findall('\\[P:.\\]', xs[0])
        if pm:
            xs[2] = ''.join(pm) + xs[2]
        header += re.sub('\\[P:.\\]', '', xs[0])
        i = 1
        while i < len(xs):
            vce, abc = xs[i:i + 2]
            id = r1.search(vce).group(1)
            if not id:
                id, vce = ('1', '[V:1]')
            vmap[id] = vmap.get(id, []) + [vce, abc]
            if id not in vorder:
                vorder[id] = i
            i += 2
    voices = []
    ixs = sorted([(i, id) for id, i in vorder.items()])
    for i, id in ixs:
        voice = ''.join(vmap[id])
        voice = fixSlurs(voice)
        voices.append((id, voice))
    return (header, voices)