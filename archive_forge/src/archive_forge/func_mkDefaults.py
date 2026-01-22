from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkDefaults(s, score, lev):
    if s.pageFmtCmd:
        s.pageFmtAbc = s.pageFmtCmd
    if not s.pageFmtAbc:
        return
    abcScale, h, w, l, r, t, b = s.pageFmtAbc
    space = abcScale * 2.117
    mils = 4 * space
    scale = 40.0 / mils
    dflts = E.Element('defaults')
    addElem(score, dflts, lev)
    scaling = E.Element('scaling')
    addElem(dflts, scaling, lev + 1)
    addElemT(scaling, 'millimeters', '%g' % mils, lev + 2)
    addElemT(scaling, 'tenths', '40', lev + 2)
    layout = E.Element('page-layout')
    addElem(dflts, layout, lev + 1)
    addElemT(layout, 'page-height', '%g' % (h * scale), lev + 2)
    addElemT(layout, 'page-width', '%g' % (w * scale), lev + 2)
    margins = E.Element('page-margins', type='both')
    addElem(layout, margins, lev + 2)
    addElemT(margins, 'left-margin', '%g' % (l * scale), lev + 3)
    addElemT(margins, 'right-margin', '%g' % (r * scale), lev + 3)
    addElemT(margins, 'top-margin', '%g' % (t * scale), lev + 3)
    addElemT(margins, 'bottom-margin', '%g' % (b * scale), lev + 3)