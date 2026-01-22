from __future__ import annotations
import functools
import itertools
import os.path
import re
import textwrap
from email.message import Message
from email.parser import Parser
from typing import Iterator
from .vendored.packaging.requirements import Requirement
def split_sections(s):
    """Split a string or iterable thereof into (section, content) pairs
    Each ``section`` is a stripped version of the section header ("[section]")
    and each ``content`` is a list of stripped lines excluding blank lines and
    comment-only lines.  If there are any such lines before the first section
    header, they're returned in a first ``section`` of ``None``.
    """
    section = None
    content = []
    for line in yield_lines(s):
        if line.startswith('['):
            if line.endswith(']'):
                if section or content:
                    yield (section, content)
                section = line[1:-1].strip()
                content = []
            else:
                raise ValueError('Invalid section heading', line)
        else:
            content.append(line)
    yield (section, content)