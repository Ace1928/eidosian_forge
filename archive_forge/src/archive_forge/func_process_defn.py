from __future__ import print_function, unicode_literals
import argparse
import collections
import io
import json
import logging
import os
import pprint
import re
import sys
import cmakelang
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.body_nodes import BodyNode
from cmakelang.parse.common import TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.funs.set import SetFnNode
def process_defn(statement):
    """
  Process one function or macro definition
  """
    semantic_tokens = statement.argtree.parg_groups[0].get_tokens(kind='semantic')
    canonical_spelling = semantic_tokens[0].spelling
    fnname = canonical_spelling.lower()
    argnames = [token.spelling.lower() for token in semantic_tokens[1:]]
    out = {'pargs': {'nargs': len(argnames)}}
    if canonical_spelling.lower() != canonical_spelling and canonical_spelling.upper() != canonical_spelling:
        out['spelling'] = canonical_spelling
    process_defn_body(statement.parent.children[1], out)
    return (fnname, out)