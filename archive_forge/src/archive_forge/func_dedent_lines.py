import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    return textwrap.dedent('\n'.join(lines)).split('\n')