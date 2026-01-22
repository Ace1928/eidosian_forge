import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds
def elementtree_indent(elem, level=0):
    """
    Recursive function to indent an ElementTree._ElementInterface
    used for pretty printing. Run indent on elem and then output
    in the normal way.

    :param elem: element to be indented. will be modified.
    :type elem: ElementTree._ElementInterface
    :param level: level of indentation for this element
    :type level: nonnegative integer
    :rtype:   ElementTree._ElementInterface
    :return:  Contents of elem indented to reflect its structure
    """
    i = '\n' + level * '  '
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + '  '
        for elem in elem:
            elementtree_indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i