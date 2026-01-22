from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def generate_cy_utility_code(self):
    arg_types = [a.type for a in self.in_definitions]
    out_types = [a.type for a in self.out_definitions]
    inline_func_decl = self.node.entry.type.declaration_code(self.node.entry.cname, pyrex=True)
    self.node.entry.used = True
    ufunc_cname = self.global_scope.next_id(self.node.entry.name + '_ufunc_def')
    will_be_called_without_gil = not (any((t.is_pyobject for t in arg_types)) or any((t.is_pyobject for t in out_types)))
    context = dict(func_cname=ufunc_cname, in_types=arg_types, out_types=out_types, inline_func_call=self.node.entry.cname, inline_func_declaration=inline_func_decl, nogil=self.node.entry.type.nogil, will_be_called_without_gil=will_be_called_without_gil)
    code = CythonUtilityCode.load('UFuncDefinition', 'UFuncs.pyx', context=context, outer_module_scope=self.global_scope)
    tree = code.get_tree(entries_only=True)
    return tree