import sys
import os.path
import re
import urllib.request, urllib.parse, urllib.error
import docutils
from docutils import nodes, utils, writers, languages, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import (unichar2tex, pick_math_environment,
def set_class_on_child(self, node, class_, index=0):
    """
        Set class `class_` on the visible child no. index of `node`.
        Do nothing if node has fewer children than `index`.
        """
    children = [n for n in node if not isinstance(n, nodes.Invisible)]
    try:
        child = children[index]
    except IndexError:
        return
    child['classes'].append(class_)