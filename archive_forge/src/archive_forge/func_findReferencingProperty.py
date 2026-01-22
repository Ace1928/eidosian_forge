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
def findReferencingProperty(node, prop, val, ids):
    global referencingProps
    if prop in referencingProps and val != '':
        if len(val) >= 7 and val[0:5] == 'url(#':
            id = val[5:val.find(')')]
            if id in ids:
                ids[id].add(node)
            else:
                ids[id] = {node}
        elif len(val) >= 8:
            id = None
            if val[0:6] == 'url("#':
                id = val[6:val.find('")')]
            elif val[0:6] == "url('#":
                id = val[6:val.find("')")]
            if id is not None:
                if id in ids:
                    ids[id].add(node)
                else:
                    ids[id] = {node}