from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
def import_module(self, module_name, module_level):
    self.imported.add(module_name)
    module_node = getsource(module_name, self.passmanager.module_dir, module_level)
    self.prefixes.append(mangle_imported_module(module_name))
    self.identifiers.append({})
    self.rename_top_level_functions(module_node)
    self.generic_visit(module_node)
    self.prefixes.pop()
    self.identifiers.pop()
    return module_node.body