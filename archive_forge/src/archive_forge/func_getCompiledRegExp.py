from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def getCompiledRegExp(self) -> re.Pattern:
    """ Return a compiled regular expression. """
    return self.compiled_re