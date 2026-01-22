from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def update_fused_defnode_entry(self, env):
    copy_attributes = ('name', 'pos', 'cname', 'func_cname', 'pyfunc_cname', 'pymethdef_cname', 'doc', 'doc_cname', 'is_member', 'scope')
    entry = self.py_func.entry
    for attr in copy_attributes:
        setattr(entry, attr, getattr(self.orig_py_func.entry, attr))
    self.py_func.name = self.orig_py_func.name
    self.py_func.doc = self.orig_py_func.doc
    env.entries.pop('__pyx_fused_cpdef', None)
    if isinstance(self.node, DefNode):
        env.entries[entry.name] = entry
    else:
        env.entries[entry.name].as_variable = entry
    env.pyfunc_entries.append(entry)
    self.py_func.entry.fused_cfunction = self
    for node in self.nodes:
        if isinstance(self.node, DefNode):
            node.fused_py_func = self.py_func
        else:
            node.py_func.fused_py_func = self.py_func
            node.entry.as_variable = entry
    self.synthesize_defnodes()
    self.stats.append(self.__signatures__)