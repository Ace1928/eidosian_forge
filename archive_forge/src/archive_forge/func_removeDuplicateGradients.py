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
def removeDuplicateGradients(doc):
    prev_num = -1
    num = 0
    referenced_ids = findReferencedElements(doc.documentElement)
    while prev_num != num:
        prev_num = num
        linear_gradients = doc.getElementsByTagName('linearGradient')
        radial_gradients = doc.getElementsByTagName('radialGradient')
        for master_id, duplicates_ids, duplicates in detect_duplicate_gradients(linear_gradients, radial_gradients):
            dedup_gradient(master_id, duplicates_ids, duplicates, referenced_ids)
            num += len(duplicates)
    return num