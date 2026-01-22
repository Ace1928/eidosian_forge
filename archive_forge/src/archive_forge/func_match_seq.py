from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
def match_seq(self, nodes, results=None) -> bool:
    return len(nodes) == 0