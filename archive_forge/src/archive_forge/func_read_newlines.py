import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def read_newlines(filename, limit=1024):
    """
    >>> tmp_path = getfixture('tmp_path')
    >>> filename = tmp_path / 'out.txt'
    >>> _ = filename.write_text('foo\\n', newline='', encoding='utf-8')
    >>> read_newlines(filename)
    '\\n'
    >>> _ = filename.write_text('foo\\r\\n', newline='', encoding='utf-8')
    >>> read_newlines(filename)
    '\\r\\n'
    >>> _ = filename.write_text('foo\\r\\nbar\\nbing\\r', newline='', encoding='utf-8')
    >>> read_newlines(filename)
    ('\\r', '\\n', '\\r\\n')
    """
    with open(filename, encoding='utf-8') as fp:
        fp.read(limit)
    return fp.newlines