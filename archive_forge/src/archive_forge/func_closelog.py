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
def closelog():
    """Close the log file."""
    global log, logfile, logfp
    logfile = ''
    if logfp:
        logfp.close()
        logfp = None
    log = initlog