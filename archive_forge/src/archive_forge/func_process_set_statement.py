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
def process_set_statement(argtree, variables):
    """
  Process a set() statement, updating the variable assignments accordingly
  """
    varname = replace_varrefs(argtree.varname.spelling, variables)
    if not argtree.value_group:
        variables.pop(varname, None)
        return
    setargs = argtree.value_group.get_tokens(kind='semantic')
    valuestr = ';'.join((arg.spelling.strip('"') for arg in setargs))
    variables[varname] = replace_varrefs(valuestr, variables)