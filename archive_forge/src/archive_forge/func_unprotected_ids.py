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
def unprotected_ids(doc, options):
    u"""Returns a list of unprotected IDs within the document doc."""
    identifiedElements = findElementsWithId(doc.documentElement)
    protectedIDs = protected_ids(identifiedElements, options)
    if protectedIDs:
        for id in protectedIDs:
            del identifiedElements[id]
    return identifiedElements