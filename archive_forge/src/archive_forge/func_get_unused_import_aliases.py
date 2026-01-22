from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import copy
import logging
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import scope
def get_unused_import_aliases(tree, sc=None):
    """Get the import aliases that aren't used.

  Arguments:
    tree: (ast.AST) An ast to find imports in.
    sc: A scope.Scope representing tree (generated from scratch if not
    provided).

  Returns:
    A list of ast.alias representing imported aliases that aren't referenced in
    the given tree.
  """
    if sc is None:
        sc = scope.analyze(tree)
    unused_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.alias):
            str_name = node.asname if node.asname is not None else node.name
            if str_name in sc.names:
                name = sc.names[str_name]
                if not name.reads:
                    unused_aliases.add(node)
            else:
                logging.warning("Imported name %s not found in scope (perhaps it's imported dynamically)", str_name)
    return unused_aliases