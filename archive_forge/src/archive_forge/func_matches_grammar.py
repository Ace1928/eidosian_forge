import ast
import sys
import warnings
from typing import Iterable, Iterator, List, Set, Tuple
from black.mode import VERSION_TO_FEATURES, Feature, TargetVersion, supports_feature
from black.nodes import syms
from blib2to3 import pygram
from blib2to3.pgen2 import driver
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from blib2to3.pgen2.tokenize import TokenError
from blib2to3.pytree import Leaf, Node
def matches_grammar(src_txt: str, grammar: Grammar) -> bool:
    drv = driver.Driver(grammar)
    try:
        drv.parse_string(src_txt, True)
    except (ParseError, TokenError, IndentationError):
        return False
    else:
        return True