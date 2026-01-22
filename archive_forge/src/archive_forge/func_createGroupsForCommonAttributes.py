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
def createGroupsForCommonAttributes(elem):
    """
    Creates <g> elements to contain runs of 3 or more
    consecutive child elements having at least one common attribute.

    Common attributes are not promoted to the <g> by this function.
    This is handled by moveCommonAttributesToParentGroup.

    If all children have a common attribute, an extra <g> is not created.

    This function acts recursively on the given element.
    """
    num = 0
    global _num_elements_removed
    for curAttr in ['clip-rule', 'display-align', 'fill', 'fill-opacity', 'fill-rule', 'font', 'font-family', 'font-size', 'font-size-adjust', 'font-stretch', 'font-style', 'font-variant', 'font-weight', 'letter-spacing', 'pointer-events', 'shape-rendering', 'stroke', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit', 'stroke-opacity', 'stroke-width', 'text-anchor', 'text-decoration', 'text-rendering', 'visibility', 'word-spacing', 'writing-mode']:
        curChild = elem.childNodes.length - 1
        while curChild >= 0:
            childNode = elem.childNodes.item(curChild)
            if childNode.nodeType == Node.ELEMENT_NODE and childNode.getAttribute(curAttr) != '' and (childNode.nodeName in ['animate', 'animateColor', 'animateMotion', 'animateTransform', 'set', 'desc', 'metadata', 'title', 'circle', 'ellipse', 'line', 'path', 'polygon', 'polyline', 'rect', 'defs', 'g', 'svg', 'symbol', 'use', 'linearGradient', 'radialGradient', 'a', 'altGlyphDef', 'clipPath', 'color-profile', 'cursor', 'filter', 'font', 'font-face', 'foreignObject', 'image', 'marker', 'mask', 'pattern', 'script', 'style', 'switch', 'text', 'view', 'animation', 'audio', 'discard', 'handler', 'listener', 'prefetch', 'solidColor', 'textArea', 'video']):
                value = childNode.getAttribute(curAttr)
                runStart, runEnd = (curChild, curChild)
                runElements = 1
                while runStart > 0:
                    nextNode = elem.childNodes.item(runStart - 1)
                    if nextNode.nodeType == Node.ELEMENT_NODE:
                        if nextNode.getAttribute(curAttr) != value:
                            break
                        else:
                            runElements += 1
                            runStart -= 1
                    else:
                        runStart -= 1
                if runElements >= 3:
                    while runEnd < elem.childNodes.length - 1:
                        if elem.childNodes.item(runEnd + 1).nodeType == Node.ELEMENT_NODE:
                            break
                        else:
                            runEnd += 1
                    runLength = runEnd - runStart + 1
                    if runLength == elem.childNodes.length:
                        if elem.nodeName == 'g' and elem.namespaceURI == NS['SVG']:
                            curChild = -1
                            continue
                    document = elem.ownerDocument
                    group = document.createElementNS(NS['SVG'], 'g')
                    group.childNodes[:] = elem.childNodes[runStart:runEnd + 1]
                    for child in group.childNodes:
                        child.parentNode = group
                    elem.childNodes[runStart:runEnd + 1] = []
                    elem.childNodes.insert(runStart, group)
                    group.parentNode = elem
                    num += 1
                    curChild = runStart - 1
                    _num_elements_removed -= 1
                else:
                    curChild -= 1
            else:
                curChild -= 1
    for childNode in elem.childNodes:
        if childNode.nodeType == Node.ELEMENT_NODE:
            num += createGroupsForCommonAttributes(childNode)
    return num