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
def moveCommonAttributesToParentGroup(elem, referencedElements):
    """
    This recursively calls this function on all children of the passed in element
    and then iterates over all child elements and removes common inheritable attributes
    from the children and places them in the parent group.  But only if the parent contains
    nothing but element children and whitespace.  The attributes are only removed from the
    children if the children are not referenced by other elements in the document.
    """
    num = 0
    childElements = []
    for child in elem.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            if not child.getAttribute('id') in referencedElements:
                childElements.append(child)
                num += moveCommonAttributesToParentGroup(child, referencedElements)
        elif child.nodeType == Node.TEXT_NODE and child.nodeValue.strip():
            return num
    if len(childElements) <= 1:
        return num
    commonAttrs = {}
    attrList = childElements[0].attributes
    for index in range(attrList.length):
        attr = attrList.item(index)
        if attr.nodeName in ['clip-rule', 'display-align', 'fill', 'fill-opacity', 'fill-rule', 'font', 'font-family', 'font-size', 'font-size-adjust', 'font-stretch', 'font-style', 'font-variant', 'font-weight', 'letter-spacing', 'pointer-events', 'shape-rendering', 'stroke', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit', 'stroke-opacity', 'stroke-width', 'text-anchor', 'text-decoration', 'text-rendering', 'visibility', 'word-spacing', 'writing-mode']:
            commonAttrs[attr.nodeName] = attr.nodeValue
    for childNum in range(len(childElements)):
        if childNum == 0:
            continue
        child = childElements[childNum]
        if child.localName in ['set', 'animate', 'animateColor', 'animateTransform', 'animateMotion']:
            continue
        distinctAttrs = []
        for name in commonAttrs:
            if child.getAttribute(name) != commonAttrs[name]:
                distinctAttrs.append(name)
        for name in distinctAttrs:
            del commonAttrs[name]
    for name in commonAttrs:
        for child in childElements:
            child.removeAttribute(name)
        elem.setAttribute(name, commonAttrs[name])
    num += (len(childElements) - 1) * len(commonAttrs)
    return num