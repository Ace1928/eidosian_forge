import hashlib
import os.path
import sys
from collections import namedtuple
from copy import copy, deepcopy
import pkgutil
from ast import literal_eval
from contextlib import suppress
from typing import List, Tuple, Union, Callable, Dict, Optional, Sequence, Generator
from .utils import bfs, logger, classify_bool, is_id_continue, is_id_start, bfs_all_unique, small_factors, OrderedSet
from .lexer import Token, TerminalDef, PatternStr, PatternRE, Pattern
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import ParsingFrontend
from .common import LexerConf, ParserConf
from .grammar import RuleOptions, Rule, Terminal, NonTerminal, Symbol, TOKEN_DEFAULT_PRIORITY
from .utils import classify, dedup_list
from .exceptions import GrammarError, UnexpectedCharacters, UnexpectedToken, ParseError, UnexpectedInput
from .tree import Tree, SlottedTree as ST
from .visitors import Transformer, Visitor, v_args, Transformer_InPlace, Transformer_NonRecursive
def resolve_term_references(term_dict):
    while True:
        changed = False
        for name, token_tree in term_dict.items():
            if token_tree is None:
                continue
            for exp in token_tree.find_data('value'):
                item, = exp.children
                if isinstance(item, NonTerminal):
                    raise GrammarError("Rules aren't allowed inside terminals (%s in %s)" % (item, name))
                elif isinstance(item, Terminal):
                    try:
                        term_value = term_dict[item.name]
                    except KeyError:
                        raise GrammarError('Terminal used but not defined: %s' % item.name)
                    assert term_value is not None
                    exp.children[0] = term_value
                    changed = True
                else:
                    assert isinstance(item, Tree)
        if not changed:
            break
    for name, term in term_dict.items():
        if term:
            for child in term.children:
                ids = [id(x) for x in child.iter_subtrees()]
                if id(term) in ids:
                    raise GrammarError("Recursion in terminal '%s' (recursion is only allowed in rules, not terminals)" % name)