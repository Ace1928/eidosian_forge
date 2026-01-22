from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import copy
import logging
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import scope
def split_import(sc, node, alias_to_remove):
    """Split an import node by moving the given imported alias into a new import.

  Arguments:
    sc: (scope.Scope) Scope computed on whole tree of the code being modified.
    node: (ast.Import|ast.ImportFrom) An import node to split.
    alias_to_remove: (ast.alias) The import alias node to remove. This must be a
      child of the given `node` argument.

  Raises:
    errors.InvalidAstError: if `node` is not appropriately contained in the tree
      represented by the scope `sc`.
  """
    parent = sc.parent(node)
    parent_list = None
    for a in ('body', 'orelse', 'finalbody'):
        if hasattr(parent, a) and node in getattr(parent, a):
            parent_list = getattr(parent, a)
            break
    else:
        raise errors.InvalidAstError('Unable to find list containing import %r on parent node %r' % (node, parent))
    idx = parent_list.index(node)
    new_import = copy.deepcopy(node)
    new_import.names = [alias_to_remove]
    node.names.remove(alias_to_remove)
    parent_list.insert(idx + 1, new_import)
    return new_import