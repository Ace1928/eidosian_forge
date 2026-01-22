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
def find_grammar_errors(text: str, start: str='start') -> List[Tuple[UnexpectedInput, str]]:
    errors = []

    def on_error(e):
        errors.append((e, _error_repr(e)))
        token_path, _ = _search_interactive_parser(e.interactive_parser.as_immutable(), lambda p: '_NL' in p.choices())
        for token_type in token_path:
            e.interactive_parser.feed_token(Token(token_type, ''))
        e.interactive_parser.feed_token(Token('_NL', '\n'))
        return True
    _tree = _get_parser().parse(text + '\n', start, on_error=on_error)
    errors_by_line = classify(errors, lambda e: e[0].line)
    errors = [el[0] for el in errors_by_line.values()]
    for e in errors:
        e[0].interactive_parser = None
    return errors