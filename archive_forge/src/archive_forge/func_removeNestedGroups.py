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
def removeNestedGroups(node):
    """
    This walks further and further down the tree, removing groups
    which do not have any attributes or a title/desc child and
    promoting their children up one level
    """
    global _num_elements_removed
    num = 0
    groupsToRemove = []
    if not (node.nodeType == Node.ELEMENT_NODE and node.nodeName == 'switch'):
        for child in node.childNodes:
            if child.nodeName == 'g' and child.namespaceURI == NS['SVG'] and (len(child.attributes) == 0):
                for grandchild in child.childNodes:
                    if grandchild.nodeType == Node.ELEMENT_NODE and grandchild.namespaceURI == NS['SVG'] and (grandchild.nodeName in ['title', 'desc']):
                        break
                else:
                    groupsToRemove.append(child)
    for g in groupsToRemove:
        while g.childNodes.length > 0:
            g.parentNode.insertBefore(g.firstChild, g)
        g.parentNode.removeChild(g)
        _num_elements_removed += 1
        num += 1
    for child in node.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            num += removeNestedGroups(child)
    return num