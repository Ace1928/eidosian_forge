from __future__ import division, print_function, unicode_literals
import re
import timeit
import codecs
import argparse
import sys
from builtins import str
from commonmark.render.html import HtmlRenderer
from commonmark.main import Parser, dumpAST
def showSpaces(t):
    t = re.sub('\\t', tabChar, t)
    t = re.sub(' ', spaceChar, t)
    t = re.sub(nbspChar, spaceChar, t)
    return t