from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mkIdentification(s, score, lev):
    if s.title:
        xs = s.title.split('\n')
        ys = '\n'.join(xs[1:])
        w = E.Element('work')
        addElem(score, w, lev + 1)
        if ys:
            addElemT(w, 'work-number', ys, lev + 2)
        addElemT(w, 'work-title', xs[0], lev + 2)
    ident = E.Element('identification')
    addElem(score, ident, lev + 1)
    for mtype, mval in s.metadata.items():
        if mtype in s.metaTypes and mtype != 'rights':
            c = E.Element('creator', type=mtype)
            c.text = mval
            addElem(ident, c, lev + 2)
    if 'rights' in s.metadata:
        c = addElemT(ident, 'rights', s.metadata['rights'], lev + 2)
    encoding = E.Element('encoding')
    addElem(ident, encoding, lev + 2)
    encoder = E.Element('encoder')
    encoder.text = 'abc2xml version %d' % VERSION
    addElem(encoding, encoder, lev + 3)
    if s.supports_tag:
        suports = E.Element('supports', attribute='new-system', element='print', type='yes', value='yes')
        addElem(encoding, suports, lev + 3)
    encodingDate = E.Element('encoding-date')
    encodingDate.text = str(datetime.date.today())
    addElem(encoding, encodingDate, lev + 3)
    s.addMeta(ident, lev + 2)