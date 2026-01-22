from pythran.passmanager import FunctionAnalysis
from pythran.tables import MODULES
import gast as ast
def is_safe_call(self, node, index):
    func_aliases = list(self.aliases[node])
    for alias in func_aliases:
        if isinstance(alias, ast.Call):
            if not self.is_safe_call(alias.args[0], index + len(alias.args) - 1):
                return False
        if alias in self.argument_effects:
            func_aes = self.argument_effects[alias]
            if func_aes[index]:
                return False
    return True