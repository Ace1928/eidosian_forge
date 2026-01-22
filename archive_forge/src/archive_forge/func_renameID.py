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
def renameID(idFrom, idTo, identifiedElements, referringNodes):
    """
    Changes the ID name from idFrom to idTo, on the declaring element
    as well as all nodes in referringNodes.

    Updates identifiedElements.

    Returns the number of bytes saved by this replacement.
    """
    num = 0
    definingNode = identifiedElements[idFrom]
    definingNode.setAttribute('id', idTo)
    num += len(idFrom) - len(idTo)
    if referringNodes is not None:
        for node in referringNodes:
            if node.nodeName == 'style' and node.namespaceURI == NS['SVG']:
                if node.firstChild is not None:
                    oldValue = ''.join((child.nodeValue for child in node.childNodes))
                    newValue = oldValue.replace('url(#' + idFrom + ')', 'url(#' + idTo + ')')
                    newValue = newValue.replace("url(#'" + idFrom + "')", 'url(#' + idTo + ')')
                    newValue = newValue.replace('url(#"' + idFrom + '")', 'url(#' + idTo + ')')
                    node.childNodes[:] = [node.ownerDocument.createTextNode(newValue)]
                    num += len(oldValue) - len(newValue)
            href = node.getAttributeNS(NS['XLINK'], 'href')
            if href == '#' + idFrom:
                node.setAttributeNS(NS['XLINK'], 'href', '#' + idTo)
                num += len(idFrom) - len(idTo)
            styles = node.getAttribute('style')
            if styles != '':
                newValue = styles.replace('url(#' + idFrom + ')', 'url(#' + idTo + ')')
                newValue = newValue.replace("url('#" + idFrom + "')", 'url(#' + idTo + ')')
                newValue = newValue.replace('url("#' + idFrom + '")', 'url(#' + idTo + ')')
                node.setAttribute('style', newValue)
                num += len(styles) - len(newValue)
            for attr in referencingProps:
                oldValue = node.getAttribute(attr)
                if oldValue != '':
                    newValue = oldValue.replace('url(#' + idFrom + ')', 'url(#' + idTo + ')')
                    newValue = newValue.replace("url('#" + idFrom + "')", 'url(#' + idTo + ')')
                    newValue = newValue.replace('url("#' + idFrom + '")', 'url(#' + idTo + ')')
                    node.setAttribute(attr, newValue)
                    num += len(oldValue) - len(newValue)
    return num