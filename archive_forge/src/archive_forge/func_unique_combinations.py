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
def unique_combinations(items, n):
    """Return `itertools.combinations`."""
    warnings.warn('docutils.utils.unique_combinations is deprecated; use itertools.combinations directly.', DeprecationWarning, stacklevel=2)
    return itertools.combinations(items, n)