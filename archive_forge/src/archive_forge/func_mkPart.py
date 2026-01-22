from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkPart(s, maten, id, lev, attrs, nstaves, rOpt):
    s.slurstack = {}
    s.glisnum = 0
    s.slidenum = 0
    s.unitLcur = s.unitL
    s.curVolta = ''
    s.lyrdash = {}
    s.linebrk = 0
    s.midprg = ['', '', '', '']
    s.gcue_on = 0
    s.gtrans = 0
    s.percVoice = 0
    s.curClef = ''
    s.nostems = 0
    s.tuning = s.tuningDef
    part = E.Element('part', id=id)
    s.overlayVnum = 0
    gstaff = s.gStaffNums.get(s.vid, 0)
    attrs_cpy = attrs.copy()
    if gstaff == 1:
        attrs_cpy['gstaff'] = nstaves
    if 'perc' in attrs_cpy.get('V', ''):
        del attrs_cpy['K']
    msre, overlay = s.mkMeasure(1, maten[0], lev + 1, attrs_cpy)
    addElem(part, msre, lev + 1)
    for i, maat in enumerate(maten[1:]):
        s.overlayVnum = s.overlayVnum + 1 if overlay else 0
        msre, next_overlay = s.mkMeasure(i + 2, maat, lev + 1)
        if overlay:
            mergePartMeasure(part, msre, s.overlayVnum, rOpt)
        else:
            addElem(part, msre, lev + 1)
        overlay = next_overlay
    return part