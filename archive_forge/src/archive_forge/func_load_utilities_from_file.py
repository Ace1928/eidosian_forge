from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
@classmethod
def load_utilities_from_file(cls, path):
    utilities = cls._utility_cache.get(path)
    if utilities:
        return utilities
    _, ext = os.path.splitext(path)
    if ext in ('.pyx', '.py', '.pxd', '.pxi'):
        comment = '#'
        strip_comments = partial(re.compile('^\\s*#(?!\\s*cython\\s*:).*').sub, '')
        rstrip = StringEncoding._unicode.rstrip
    else:
        comment = '/'
        strip_comments = partial(re.compile('^\\s*//.*|/\\*[^*]*\\*/').sub, '')
        rstrip = partial(re.compile('\\s+(\\\\?)$').sub, '\\1')
    match_special = re.compile('^%(C)s{5,30}\\s*(?P<name>(?:\\w|\\.)+)\\s*%(C)s{5,30}|^%(C)s+@(?P<tag>\\w+)\\s*:\\s*(?P<value>(?:\\w|[.:])+)' % {'C': comment}).match
    match_type = re.compile('(.+)[.](proto(?:[.]\\S+)?|impl|init|cleanup)$').match
    all_lines = read_utilities_hook(path)
    utilities = defaultdict(lambda: [None, None, {}])
    lines = []
    tags = defaultdict(set)
    utility = type = None
    begin_lineno = 0
    for lineno, line in enumerate(all_lines):
        m = match_special(line)
        if m:
            if m.group('name'):
                cls._add_utility(utility, type, lines, begin_lineno, tags)
                begin_lineno = lineno + 1
                del lines[:]
                tags.clear()
                name = m.group('name')
                mtype = match_type(name)
                if mtype:
                    name, type = mtype.groups()
                else:
                    type = 'impl'
                utility = utilities[name]
            else:
                tags[m.group('tag')].add(m.group('value'))
                lines.append('')
        else:
            lines.append(rstrip(strip_comments(line)))
    if utility is None:
        raise ValueError('Empty utility code file')
    cls._add_utility(utility, type, lines, begin_lineno, tags)
    utilities = dict(utilities)
    cls._utility_cache[path] = utilities
    return utilities