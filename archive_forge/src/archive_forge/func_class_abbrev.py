import html
import re
from collections import defaultdict
def class_abbrev(type):
    """
    Abbreviate an NE class name.
    :type type: str
    :rtype: str
    """
    try:
        return long2short[type]
    except KeyError:
        return type