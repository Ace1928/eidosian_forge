from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def doc_to_help(doc):
    """Takes a __doc__ string and reformats it as help."""
    doc = doc.strip()
    whitespace_only_line = re.compile('^[ \t]+$', re.M)
    doc = whitespace_only_line.sub('', doc)
    doc = trim_docstring(doc)
    doc = re.sub('(?<=\\S)\\n(?=\\S)', ' ', doc, flags=re.M)
    return doc