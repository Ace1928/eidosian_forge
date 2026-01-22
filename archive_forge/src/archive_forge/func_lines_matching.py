from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
def lines_matching(self, *regexes: str) -> set[TLineNo]:
    """Find the lines matching one of a list of regexes.

        Returns a set of line numbers, the lines that contain a match for one
        of the regexes in `regexes`.  The entire line needn't match, just a
        part of it.

        """
    combined = join_regex(regexes)
    regex_c = re.compile(combined)
    matches = set()
    for i, ltext in enumerate(self.lines, start=1):
        if regex_c.search(ltext):
            matches.add(i)
    return matches