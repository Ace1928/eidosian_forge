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
def read_lines(self):
    """Internal: read lines until EOF or outerboundary."""
    if self._binary_file:
        self.file = self.__file = BytesIO()
    else:
        self.file = self.__file = StringIO()
    if self.outerboundary:
        self.read_lines_to_outerboundary()
    else:
        self.read_lines_to_eof()