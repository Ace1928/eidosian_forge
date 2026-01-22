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
def computeGradientBucketKey(grad):
    gradBucketAttr = ['gradientUnits', 'spreadMethod', 'gradientTransform', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'fx', 'fy', 'r']
    gradStopBucketsAttr = ['offset', 'stop-color', 'stop-opacity', 'style']
    subKeys = [grad.getAttribute(a) for a in gradBucketAttr]
    subKeys.append(grad.getAttributeNS(NS['XLINK'], 'href'))
    stops = grad.getElementsByTagName('stop')
    if stops.length:
        for i in range(stops.length):
            stop = stops.item(i)
            for attr in gradStopBucketsAttr:
                stopKey = stop.getAttribute(attr)
                subKeys.append(stopKey)
    return '\x1e'.join(subKeys)