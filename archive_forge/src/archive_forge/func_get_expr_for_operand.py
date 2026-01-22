from sympy.external import import_module
import os
def get_expr_for_operand(self, combined_variable):
    """Gives out SymPy Codegen AST node

            AST node returned is corresponding to
            combined variable passed.Combined variable contains
            variable content and type of variable

            """
    if combined_variable[1] == 'identifier':
        return Symbol(combined_variable[0])
    if combined_variable[1] == 'literal':
        if '.' in combined_variable[0]:
            return Float(float(combined_variable[0]))
        else:
            return Integer(int(combined_variable[0]))
    if combined_variable[1] == 'expr':
        return combined_variable[0]
    if combined_variable[1] == 'boolean':
        return true if combined_variable[0] == 'true' else false