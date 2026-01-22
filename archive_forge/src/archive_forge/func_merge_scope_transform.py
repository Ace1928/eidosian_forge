from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
def merge_scope_transform(module_node):
    module_node.scope.merge_in(scope)
    return module_node