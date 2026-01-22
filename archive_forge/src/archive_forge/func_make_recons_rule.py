import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
def make_recons_rule(origin, expansion, old_expansion):
    return Rule(origin, expansion, alias=_MakeTreeMatch(origin.name, old_expansion))