from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def scourUnitlessLength(length, renderer_workaround=False, is_control_point=False):
    """
    Scours the numeric part of a length only. Does not accept units.

    This is faster than scourLength on elements guaranteed not to
    contain units.
    """
    if not isinstance(length, Decimal):
        length = getcontext().create_decimal(str(length))
    initial_length = length
    if is_control_point:
        length = scouringContextC.plus(length)
    else:
        length = scouringContext.plus(length)
    intLength = length.to_integral_value()
    if length == intLength:
        length = Decimal(intLength)
    else:
        length = length.normalize()
    nonsci = '{0:f}'.format(length)
    nonsci = '{0:f}'.format(initial_length.quantize(Decimal(nonsci)))
    if not renderer_workaround:
        if len(nonsci) > 2 and nonsci[:2] == '0.':
            nonsci = nonsci[1:]
        elif len(nonsci) > 3 and nonsci[:3] == '-0.':
            nonsci = '-' + nonsci[2:]
    return_value = nonsci
    if len(nonsci) > 3:
        exponent = length.adjusted()
        length = length.scaleb(-exponent).normalize()
        sci = six.text_type(length) + 'e' + six.text_type(exponent)
        if len(sci) < len(nonsci):
            return_value = sci
    return return_value