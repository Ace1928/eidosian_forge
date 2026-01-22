from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def copy_def(self, env):
    """
        Create a copy of the original def or lambda function for specialized
        versions.
        """
    fused_compound_types = PyrexTypes.unique([arg.type for arg in self.node.args if arg.type.is_fused])
    fused_types = self._get_fused_base_types(fused_compound_types)
    permutations = PyrexTypes.get_all_specialized_permutations(fused_types)
    self.fused_compound_types = fused_compound_types
    if self.node.entry in env.pyfunc_entries:
        env.pyfunc_entries.remove(self.node.entry)
    for cname, fused_to_specific in permutations:
        copied_node = copy.deepcopy(self.node)
        copied_node.entry.signature = self.node.entry.signature
        self._specialize_function_args(copied_node.args, fused_to_specific)
        copied_node.return_type = self.node.return_type.specialize(fused_to_specific)
        copied_node.analyse_declarations(env)
        self.create_new_local_scope(copied_node, env, fused_to_specific)
        self.specialize_copied_def(copied_node, cname, self.node.entry, fused_to_specific, fused_compound_types)
        PyrexTypes.specialize_entry(copied_node.entry, cname)
        copied_node.entry.used = True
        env.entries[copied_node.entry.name] = copied_node.entry
        if not self.replace_fused_typechecks(copied_node):
            break
    self.orig_py_func = self.node
    self.py_func = self.make_fused_cpdef(self.node, env, is_def=True)