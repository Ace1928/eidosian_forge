import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
def parse_rulename(s):
    """Parse rule names that may contain a template syntax (like rule{a, b, ...})"""
    name, args_str = re.match('(\\w+)(?:{(.+)})?', s).groups()
    args = args_str and [a.strip() for a in args_str.split(',')]
    return (name, args)