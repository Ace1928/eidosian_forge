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
def repairStyle(node, options):
    num = 0
    styleMap = _getStyle(node)
    if styleMap:
        for prop in ['fill', 'stroke']:
            if prop in styleMap:
                chunk = styleMap[prop].split(') ')
                if len(chunk) == 2 and (chunk[0][:5] == 'url(#' or chunk[0][:6] == 'url("#' or chunk[0][:6] == "url('#") and (chunk[1] == 'rgb(0, 0, 0)'):
                    styleMap[prop] = chunk[0] + ')'
                    num += 1
        if 'opacity' in styleMap:
            opacity = float(styleMap['opacity'])
            if opacity == 0.0:
                for uselessStyle in ['fill', 'fill-opacity', 'fill-rule', 'stroke', 'stroke-linejoin', 'stroke-opacity', 'stroke-miterlimit', 'stroke-linecap', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-opacity']:
                    if uselessStyle in styleMap and (not styleInheritedByChild(node, uselessStyle)):
                        del styleMap[uselessStyle]
                        num += 1
        if 'stroke' in styleMap and styleMap['stroke'] == 'none':
            for strokestyle in ['stroke-width', 'stroke-linejoin', 'stroke-miterlimit', 'stroke-linecap', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-opacity']:
                if strokestyle in styleMap and (not styleInheritedByChild(node, strokestyle)):
                    del styleMap[strokestyle]
                    num += 1
            if not styleInheritedByChild(node, 'stroke'):
                if styleInheritedFromParent(node, 'stroke') in [None, 'none']:
                    del styleMap['stroke']
                    num += 1
        if 'fill' in styleMap and styleMap['fill'] == 'none':
            for fillstyle in ['fill-rule', 'fill-opacity']:
                if fillstyle in styleMap and (not styleInheritedByChild(node, fillstyle)):
                    del styleMap[fillstyle]
                    num += 1
        if 'fill-opacity' in styleMap:
            fillOpacity = float(styleMap['fill-opacity'])
            if fillOpacity == 0.0:
                for uselessFillStyle in ['fill', 'fill-rule']:
                    if uselessFillStyle in styleMap and (not styleInheritedByChild(node, uselessFillStyle)):
                        del styleMap[uselessFillStyle]
                        num += 1
        if 'stroke-opacity' in styleMap:
            strokeOpacity = float(styleMap['stroke-opacity'])
            if strokeOpacity == 0.0:
                for uselessStrokeStyle in ['stroke', 'stroke-width', 'stroke-linejoin', 'stroke-linecap', 'stroke-dasharray', 'stroke-dashoffset']:
                    if uselessStrokeStyle in styleMap and (not styleInheritedByChild(node, uselessStrokeStyle)):
                        del styleMap[uselessStrokeStyle]
                        num += 1
        if 'stroke-width' in styleMap:
            strokeWidth = SVGLength(styleMap['stroke-width'])
            if strokeWidth.value == 0.0:
                for uselessStrokeStyle in ['stroke', 'stroke-linejoin', 'stroke-linecap', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-opacity']:
                    if uselessStrokeStyle in styleMap and (not styleInheritedByChild(node, uselessStrokeStyle)):
                        del styleMap[uselessStrokeStyle]
                        num += 1
        if not mayContainTextNodes(node):
            for fontstyle in ['font-family', 'font-size', 'font-stretch', 'font-size-adjust', 'font-style', 'font-variant', 'font-weight', 'letter-spacing', 'line-height', 'kerning', 'text-align', 'text-anchor', 'text-decoration', 'text-rendering', 'unicode-bidi', 'word-spacing', 'writing-mode']:
                if fontstyle in styleMap:
                    del styleMap[fontstyle]
                    num += 1
        for inkscapeStyle in ['-inkscape-font-specification']:
            if inkscapeStyle in styleMap:
                del styleMap[inkscapeStyle]
                num += 1
        if 'overflow' in styleMap:
            if node.nodeName not in ['svg', 'symbol', 'image', 'foreignObject', 'marker', 'pattern']:
                del styleMap['overflow']
                num += 1
            elif node != node.ownerDocument.documentElement:
                if styleMap['overflow'] == 'hidden':
                    del styleMap['overflow']
                    num += 1
            elif styleMap['overflow'] == 'visible':
                del styleMap['overflow']
                num += 1
        if options.style_to_xml:
            for propName in list(styleMap):
                if propName in svgAttributes:
                    node.setAttribute(propName, styleMap[propName])
                    del styleMap[propName]
        _setStyle(node, styleMap)
    for child in node.childNodes:
        num += repairStyle(child, options)
    return num