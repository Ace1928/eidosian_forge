from sympy.external import import_module
import os
def transform_compound_assignment_operator(self, node):
    """Transformation function for handling shorthand operators

            Returns
            =======

            augmented_assignment_expression: Codegen AST node
                    shorthand assignment expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If the shorthand operator for bitwise operators
                (~=, ^=, &=, |=, <<=, >>=) is encountered

            """
    return self.transform_binary_operator(node)