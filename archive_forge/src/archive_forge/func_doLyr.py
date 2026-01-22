from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doLyr(s, n, nt, lev):
    for i, lyrobj in enumerate(n.objs):
        lyrel = E.Element('lyric', number=str(i + 1))
        if lyrobj.name == 'syl':
            dash = len(lyrobj.t) == 2
            if dash:
                if i in s.lyrdash:
                    type = 'middle'
                else:
                    type = 'begin'
                    s.lyrdash[i] = 1
            elif i in s.lyrdash:
                type = 'end'
                del s.lyrdash[i]
            else:
                type = 'single'
            addElemT(lyrel, 'syllabic', type, lev + 1)
            txt = lyrobj.t[0]
            txt = re.sub('(?<!\\\\)~', ' ', txt)
            txt = re.sub('\\\\(.)', '\\1', txt)
            addElemT(lyrel, 'text', txt, lev + 1)
        elif lyrobj.name == 'ext' and i in s.prevLyric:
            pext = s.prevLyric[i].find('extend')
            if pext == None:
                ext = E.Element('extend', type='start')
                addElem(s.prevLyric[i], ext, lev + 1)
            elif pext.get('type') == 'stop':
                pext.set('type', 'continue')
            ext = E.Element('extend', type='stop')
            addElem(lyrel, ext, lev + 1)
        elif lyrobj.name == 'ext':
            info('lyric extend error')
            continue
        else:
            continue
        addElem(nt, lyrel, lev)
        s.prevLyric[i] = lyrel