import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def toc_sort(entry):
    """Sort the TOC by order of appearance in text"""
    return re.search('^<(h%d).*?id=(["\\\'])%s\\2.*>%s</\\1>$' % (entry[0], entry[1], re.escape(entry[2])), text, re.M).start()