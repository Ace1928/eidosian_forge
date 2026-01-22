import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def get_all_nodes(self):
    nodes = []
    for subgraph in self.get_subgraphs():
        nodes.extend(subgraph.get_all_nodes())
    nodes.extend(self._nodes)
    return nodes