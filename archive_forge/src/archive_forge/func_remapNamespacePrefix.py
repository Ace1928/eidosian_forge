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
def remapNamespacePrefix(node, oldprefix, newprefix):
    if node is None or node.nodeType != Node.ELEMENT_NODE:
        return
    if node.prefix == oldprefix:
        localName = node.localName
        namespace = node.namespaceURI
        doc = node.ownerDocument
        parent = node.parentNode
        newNode = None
        if newprefix != '':
            newNode = doc.createElementNS(namespace, newprefix + ':' + localName)
        else:
            newNode = doc.createElement(localName)
        attrList = node.attributes
        for i in range(attrList.length):
            attr = attrList.item(i)
            newNode.setAttributeNS(attr.namespaceURI, attr.name, attr.nodeValue)
        for child in node.childNodes:
            newNode.appendChild(child.cloneNode(True))
        parent.replaceChild(newNode, node)
        node = newNode
    for child in node.childNodes:
        remapNamespacePrefix(child, oldprefix, newprefix)