import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def split_escaped_whitespace(text):
    """
    Split `text` on escaped whitespace (null+space or null+newline).
    Return a list of strings.
    """
    strings = text.split('\x00 ')
    strings = [string.split('\x00\n') for string in strings]
    return list(itertools.chain(*strings))