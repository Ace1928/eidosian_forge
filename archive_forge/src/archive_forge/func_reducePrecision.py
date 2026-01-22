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
def reducePrecision(element):
    """
    Because opacities, letter spacings, stroke widths and all that don't need
    to be preserved in SVG files with 9 digits of precision.

    Takes all of these attributes, in the given element node and its children,
    and reduces their precision to the current Decimal context's precision.
    Also checks for the attributes actually being lengths, not 'inherit', 'none'
    or anything that isn't an SVGLength.

    Returns the number of bytes saved after performing these reductions.
    """
    num = 0
    styles = _getStyle(element)
    for lengthAttr in ['opacity', 'flood-opacity', 'fill-opacity', 'stroke-opacity', 'stop-opacity', 'stroke-miterlimit', 'stroke-dashoffset', 'letter-spacing', 'word-spacing', 'kerning', 'font-size-adjust', 'font-size', 'stroke-width']:
        val = element.getAttribute(lengthAttr)
        if val != '':
            valLen = SVGLength(val)
            if valLen.units != Unit.INVALID:
                newVal = scourLength(val)
                if len(newVal) < len(val):
                    num += len(val) - len(newVal)
                    element.setAttribute(lengthAttr, newVal)
        if lengthAttr in styles:
            val = styles[lengthAttr]
            valLen = SVGLength(val)
            if valLen.units != Unit.INVALID:
                newVal = scourLength(val)
                if len(newVal) < len(val):
                    num += len(val) - len(newVal)
                    styles[lengthAttr] = newVal
    _setStyle(element, styles)
    for child in element.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            num += reducePrecision(child)
    return num