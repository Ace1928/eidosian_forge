from sympy.external import import_module
import os
def transform_character_literal(self, node):
    """Transformation function for character literal

            Used to get the value of the given character literal.

            Returns
            =======

            val : int
                val contains the ascii value of the character literal

            Notes
            =====

            Only for cases where character is assigned to a integer value,
            since character literal is not in SymPy AST

            """
    try:
        value = next(node.get_tokens()).spelling
    except (StopIteration, ValueError):
        value = node.literal
    return ord(str(value[1]))