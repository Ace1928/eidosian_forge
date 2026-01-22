from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
import xml.etree.ElementTree as etree
def detab(self, block: str) -> str:
    """ Remove one level of indent from a block.

        Preserve lazily indented blocks by only removing indent from indented lines.
        """
    lines = block.split('\n')
    for i, line in enumerate(lines):
        if line.startswith(' ' * 4):
            lines[i] = line[4:]
    return '\n'.join(lines)