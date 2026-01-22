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
def removeDefaultAttributeValues(node, options, tainted=set()):
    u"""'tainted' keeps a set of attributes defined in parent nodes.

    For such attributes, we don't delete attributes with default values."""
    num = 0
    if node.nodeType != Node.ELEMENT_NODE:
        return 0
    for attribute in default_attributes_universal:
        num += removeDefaultAttributeValue(node, attribute)
    if node.nodeName in default_attributes_per_element:
        for attribute in default_attributes_per_element[node.nodeName]:
            num += removeDefaultAttributeValue(node, attribute)
    attributes = [node.attributes.item(i).nodeName for i in range(node.attributes.length)]
    for attribute in attributes:
        if attribute not in tainted:
            if attribute in default_properties:
                if node.getAttribute(attribute) == default_properties[attribute]:
                    node.removeAttribute(attribute)
                    num += 1
                else:
                    tainted = taint(tainted, attribute)
    styles = _getStyle(node)
    for attribute in list(styles):
        if attribute not in tainted:
            if attribute in default_properties:
                if styles[attribute] == default_properties[attribute]:
                    del styles[attribute]
                    num += 1
                else:
                    tainted = taint(tainted, attribute)
    _setStyle(node, styles)
    for child in node.childNodes:
        num += removeDefaultAttributeValues(child, options, tainted.copy())
    return num