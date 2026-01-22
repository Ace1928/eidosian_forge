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
def compute_id_lengths(highest):
    """Compute how many IDs are available of a given size

    Example:
        >>> lengths = list(compute_id_lengths(512))
        >>> lengths
        [(1, 26), (2, 676)]
        >>> total_limit = sum(x[1] for x in lengths)
        >>> total_limit
        702
        >>> intToID(total_limit, '')
        'zz'

    Which tells us that we got 26 IDs of length 1 and up to 676 IDs of length two
    if we need to allocate 512 IDs.

    :param highest: Highest ID that need to be allocated
    :return: An iterator that returns tuples of (id-length, use-limit).  The
     use-limit applies only to the given id-length (i.e. it is excluding IDs
     of shorter length).  Note that the sum of the use-limit values is always
     equal to or greater than the highest param.
    """
    step = 26
    id_length = 0
    use_limit = 1
    while highest:
        id_length += 1
        use_limit *= step
        yield (id_length, use_limit)
        highest = int((highest - 1) / step)