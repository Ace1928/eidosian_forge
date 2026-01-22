from __future__ import unicode_literals
import argparse
import collections
import io
import json
import logging
import os
import shutil
import sys
import cmakelang
from cmakelang import common
from cmakelang import configuration
from cmakelang import config_util
from cmakelang.format import formatter
from cmakelang import lex
from cmakelang import markup
from cmakelang import parse
from cmakelang.parse.argument_nodes import StandardParser2
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.printer import dump_tree as dump_parse
from cmakelang.parse.funs import standard_funs
def dump_markup(nodes, config, outfile=None, indent=None):
    """
  Print a tree of node objects for debugging purposes. Takes as input a full
  parse tree.
  """
    if indent is None:
        indent = ''
    if outfile is None:
        outfile = sys.stdout
    for idx, node in enumerate(nodes):
        if not isinstance(node, TreeNode):
            continue
        noderep = repr(node)
        if sys.version_info[0] < 3:
            noderep = getattr(noderep, 'decode')('utf-8')
        if node.node_type is NodeType.COMMENT:
            outfile.write(noderep)
            outfile.write('\n')
            inlines = []
            for token in node.children:
                assert isinstance(token, lex.Token)
                if token.type == lex.TokenType.COMMENT:
                    inlines.append(token.spelling.strip().lstrip('#'))
            items = markup.parse(inlines, config)
            for item in items:
                outfile.write('{}\n'.format(item))
            outfile.write('\n')
        if not hasattr(node, 'children'):
            continue
        if idx + 1 == len(nodes):
            dump_markup(node.children, config, outfile, indent + '    ')
        else:
            dump_markup(node.children, config, outfile, indent + '│   ')