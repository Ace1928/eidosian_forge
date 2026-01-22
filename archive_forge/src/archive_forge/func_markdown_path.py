import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def markdown_path(path, encoding='utf-8', html4tags=False, tab_width=DEFAULT_TAB_WIDTH, safe_mode=None, extras=None, link_patterns=None, footnote_title=None, footnote_return_symbol=None, use_file_vars=False):
    fp = codecs.open(path, 'r', encoding)
    text = fp.read()
    fp.close()
    return Markdown(html4tags=html4tags, tab_width=tab_width, safe_mode=safe_mode, extras=extras, link_patterns=link_patterns, footnote_title=footnote_title, footnote_return_symbol=footnote_return_symbol, use_file_vars=use_file_vars).convert(text)