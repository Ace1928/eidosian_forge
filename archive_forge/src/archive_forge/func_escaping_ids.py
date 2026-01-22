from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
def escaping_ids(self, scope_stmt, stmts):
    """gather sets of identifiers defined in stmts and used out of it"""
    assigned_nodes = self.gather(IsAssigned, self.make_fake(stmts))
    escaping = set()
    for assigned_node in assigned_nodes:
        head = self.def_use_chains.chains[assigned_node]
        for user in head.users():
            if scope_stmt not in self.ancestors[user.node]:
                escaping.add(head.name())
    return escaping