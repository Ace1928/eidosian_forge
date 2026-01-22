from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def getInOut(options):
    if options.infilename:
        infile = maybe_gziped_file(options.infilename, 'rb')
    else:
        try:
            infile = sys.stdin.buffer
        except AttributeError:
            infile = sys.stdin
        if sys.stdin.isatty():
            _options_parser.error('No input file specified, see --help for detailed usage information')
    if options.outfilename:
        outfile = maybe_gziped_file(options.outfilename, 'wb')
    else:
        try:
            outfile = sys.stdout.buffer
        except AttributeError:
            outfile = sys.stdout
        options.stdout = sys.stderr
    return [infile, outfile]