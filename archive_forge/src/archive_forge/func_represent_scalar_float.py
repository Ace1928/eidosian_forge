from __future__ import print_function, absolute_import, division
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.compat import text_type, binary_type, to_unicode, PY2, PY3, ordereddict
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.scalarstring import (
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
import datetime
import sys
import types
from ruamel.yaml.comments import (
def represent_scalar_float(self, data):
    """ this is way more complicated """
    value = None
    anchor = data.yaml_anchor(any=True)
    if data != data or (data == 0.0 and data == 1.0):
        value = u'.nan'
    elif data == self.inf_value:
        value = u'.inf'
    elif data == -self.inf_value:
        value = u'-.inf'
    if value:
        return self.represent_scalar(u'tag:yaml.org,2002:float', value, anchor=anchor)
    if data._exp is None and data._prec > 0 and (data._prec == data._width - 1):
        value = u'{}{:d}.'.format(data._m_sign if data._m_sign else '', abs(int(data)))
    elif data._exp is None:
        prec = data._prec
        ms = data._m_sign if data._m_sign else ''
        value = u'{}{:0{}.{}f}'.format(ms, abs(data), data._width - len(ms), data._width - prec - 1)
        if prec == 0 or (prec == 1 and ms != ''):
            value = value.replace(u'0.', u'.')
        while len(value) < data._width:
            value += u'0'
    else:
        m, es = u'{:{}.{}e}'.format(data, data._width, data._width + (1 if data._m_sign else 0)).split('e')
        w = data._width if data._prec > 0 else data._width + 1
        if data < 0:
            w += 1
        m = m[:w]
        e = int(es)
        m1, m2 = m.split('.')
        while len(m1) + len(m2) < data._width - (1 if data._prec >= 0 else 0):
            m2 += u'0'
        if data._m_sign and data > 0:
            m1 = '+' + m1
        esgn = u'+' if data._e_sign else ''
        if data._prec < 0:
            if m2 != u'0':
                e -= len(m2)
            else:
                m2 = ''
            while len(m1) + len(m2) - (1 if data._m_sign else 0) < data._width:
                m2 += u'0'
                e -= 1
            value = m1 + m2 + data._exp + u'{:{}0{}d}'.format(e, esgn, data._e_width)
        elif data._prec == 0:
            e -= len(m2)
            value = m1 + m2 + u'.' + data._exp + u'{:{}0{}d}'.format(e, esgn, data._e_width)
        else:
            if data._m_lead0 > 0:
                m2 = u'0' * (data._m_lead0 - 1) + m1 + m2
                m1 = u'0'
                m2 = m2[:-data._m_lead0]
                e += data._m_lead0
            while len(m1) < data._prec:
                m1 += m2[0]
                m2 = m2[1:]
                e -= 1
            value = m1 + u'.' + m2 + data._exp + u'{:{}0{}d}'.format(e, esgn, data._e_width)
    if value is None:
        value = to_unicode(repr(data)).lower()
    return self.represent_scalar(u'tag:yaml.org,2002:float', value, anchor=anchor)