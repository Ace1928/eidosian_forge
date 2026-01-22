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
def convertColors(element):
    """
       Recursively converts all color properties into #RRGGBB format if shorter
    """
    numBytes = 0
    if element.nodeType != Node.ELEMENT_NODE:
        return 0
    attrsToConvert = []
    if element.nodeName in ['rect', 'circle', 'ellipse', 'polygon', 'line', 'polyline', 'path', 'g', 'a']:
        attrsToConvert = ['fill', 'stroke']
    elif element.nodeName in ['stop']:
        attrsToConvert = ['stop-color']
    elif element.nodeName in ['solidColor']:
        attrsToConvert = ['solid-color']
    styles = _getStyle(element)
    for attr in attrsToConvert:
        oldColorValue = element.getAttribute(attr)
        if oldColorValue != '':
            newColorValue = convertColor(oldColorValue)
            oldBytes = len(oldColorValue)
            newBytes = len(newColorValue)
            if oldBytes > newBytes:
                element.setAttribute(attr, newColorValue)
                numBytes += oldBytes - len(element.getAttribute(attr))
        if attr in styles:
            oldColorValue = styles[attr]
            newColorValue = convertColor(oldColorValue)
            oldBytes = len(oldColorValue)
            newBytes = len(newColorValue)
            if oldBytes > newBytes:
                styles[attr] = newColorValue
                numBytes += oldBytes - newBytes
    _setStyle(element, styles)
    for child in element.childNodes:
        numBytes += convertColors(child)
    return numBytes