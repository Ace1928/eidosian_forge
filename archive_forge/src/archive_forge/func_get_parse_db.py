import importlib
import logging
from cmakelang import lex
from cmakelang.parse.additional_nodes import ShellCommandNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, KwargBreaker, TreeNode
from cmakelang.parse.util import (
from cmakelang.parse.funs import standard_funs
def get_parse_db():
    """
  Returns a dictionary mapping statement name to parse functor for that
  statement.
  """
    parse_db = {}
    for subname in SUBMODULE_NAMES:
        submodule = importlib.import_module('cmakelang.parse.funs.' + subname)
        submodule.populate_db(parse_db)
    for key in ('if', 'else', 'elseif', 'endif', 'while', 'endwhile'):
        parse_db[key] = ConditionalGroupNode.parse
    for key in ('function', 'macro'):
        parse_db[key] = StandardParser('1+')
    for key in ('endfunction', 'endmacro'):
        parse_db[key] = StandardParser('?')
    parse_db.update(get_funtree(standard_funs.get_fn_spec()))
    return parse_db