from __future__ import absolute_import
from lark.exceptions import ConfigurationError, assert_config
import sys, os, pickle, hashlib
from io import open
import tempfile
from warnings import warn
from .utils import STRING_TYPE, Serialize, SerializeMemoizer, FS, isascii, logger, ABC, abstractmethod
from .load_grammar import load_grammar, FromPackageLoader, Grammar, verify_used_files
from .tree import Tree
from .common import LexerConf, ParserConf
from .lexer import Lexer, TraditionalLexer, TerminalDef, LexerThread
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import get_frontend, _get_lexer_callbacks
from .grammar import Rule
import re
@grammar_source.setter
def grammar_source(self, value):
    self.source_grammar = value