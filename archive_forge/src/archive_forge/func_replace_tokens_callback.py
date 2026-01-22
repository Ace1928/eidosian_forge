import csv
import functools
import logging
import re
from importlib import import_module
from humanfriendly.compat import StringIO
from humanfriendly.text import dedent, split_paragraphs, trim_empty_lines
def replace_tokens_callback(match, meta_variables, replace_fn):
    token = match.group(0)
    if not (re.match('^[A-Z][A-Z0-9_]+$', token) and token not in meta_variables):
        token = replace_fn(token)
    return token