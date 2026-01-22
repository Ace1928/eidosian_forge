import inspect
import keyword
import linecache
import os
import pydoc
import sys
import tempfile
import time
import tokenize
import traceback
import warnings
from html import escape as html_escape
def grey(text):
    if text:
        return '<font color="#909090">' + text + '</font>'
    else:
        return ''