import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen
from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated
def discard_line(self):
    if self.linebuffer and len(self.linebuffer) > 1:
        line = self.linebuffer.pop(0)
        self._rewind_numchars += len(line)
    else:
        self.stream.readline()