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
def removeDefaultAttributeValue(node, attribute):
    """
    Removes the DefaultAttribute 'attribute' from 'node' if specified conditions are fulfilled

    Warning: Does NOT check if the attribute is actually valid for the passed element type for increased preformance!
    """
    if not node.hasAttribute(attribute.name):
        return 0
    if isinstance(attribute.value, str):
        if node.getAttribute(attribute.name) == attribute.value:
            if attribute.conditions is None or attribute.conditions(node):
                node.removeAttribute(attribute.name)
                return 1
    else:
        nodeValue = SVGLength(node.getAttribute(attribute.name))
        if attribute.value is None or (nodeValue.value == attribute.value and (not nodeValue.units == Unit.INVALID)):
            if attribute.units is None or nodeValue.units == attribute.units or (isinstance(attribute.units, list) and nodeValue.units in attribute.units):
                if attribute.conditions is None or attribute.conditions(node):
                    node.removeAttribute(attribute.name)
                    return 1
    return 0