import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
def get_column_width(self):
    """Return columnwidth for current cell (not multicell)."""
    try:
        return '%.2f\\DUtablewidth' % self._col_width[self._cell_in_row]
    except IndexError:
        return '*'