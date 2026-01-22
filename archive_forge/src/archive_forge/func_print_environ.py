from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
def print_environ(environ=os.environ):
    """Dump the shell environment as HTML."""
    keys = sorted(environ.keys())
    print()
    print('<H3>Shell Environment:</H3>')
    print('<DL>')
    for key in keys:
        print('<DT>', html.escape(key), '<DD>', html.escape(environ[key]))
    print('</DL>')
    print()