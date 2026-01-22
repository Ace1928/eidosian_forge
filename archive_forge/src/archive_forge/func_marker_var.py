import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
def marker_var(remaining):
    m = IDENTIFIER.match(remaining)
    if m:
        result = m.groups()[0]
        remaining = remaining[m.end():]
    elif not remaining:
        raise SyntaxError('unexpected end of input')
    else:
        q = remaining[0]
        if q not in '\'"':
            raise SyntaxError('invalid expression: %s' % remaining)
        oq = '\'"'.replace(q, '')
        remaining = remaining[1:]
        parts = [q]
        while remaining:
            if remaining[0] == q:
                break
            elif remaining[0] == oq:
                parts.append(oq)
                remaining = remaining[1:]
            else:
                m = STRING_CHUNK.match(remaining)
                if not m:
                    raise SyntaxError('error in string literal: %s' % remaining)
                parts.append(m.groups()[0])
                remaining = remaining[m.end():]
        else:
            s = ''.join(parts)
            raise SyntaxError('unterminated string: %s' % s)
        parts.append(q)
        result = ''.join(parts)
        remaining = remaining[1:].lstrip()
    return (result, remaining)