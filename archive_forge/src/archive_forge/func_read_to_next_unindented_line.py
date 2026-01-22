import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def read_to_next_unindented_line(self):

    def is_unindented(line):
        return line.strip() and len(line.lstrip()) == len(line)
    return self.read_to_condition(is_unindented)