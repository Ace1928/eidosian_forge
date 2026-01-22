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
def process_break_exits(self, exits: set[ArcStart]) -> None:
    """Add arcs due to jumps from `exits` being breaks."""
    for block in self.nearest_blocks():
        if block.process_break_exits(exits, self.add_arc):
            break