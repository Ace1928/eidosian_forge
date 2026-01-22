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
def removeNamespacedAttributes(node, namespaces):
    num = 0
    if node.nodeType == Node.ELEMENT_NODE:
        attrList = node.attributes
        attrsToRemove = []
        for attrNum in range(attrList.length):
            attr = attrList.item(attrNum)
            if attr is not None and attr.namespaceURI in namespaces:
                attrsToRemove.append(attr.nodeName)
        for attrName in attrsToRemove:
            node.removeAttribute(attrName)
        num += len(attrsToRemove)
        for child in node.childNodes:
            num += removeNamespacedAttributes(child, namespaces)
    return num