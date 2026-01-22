from sympy.external import import_module
import os
def transform_while_stmt(self, node):
    """Transformation function for handling while statement

            Returns
            =======

            while statement : Codegen AST Node
                contains the while statement node having condition and
                statement block

            """
    children = node.get_children()
    condition = self.transform(next(children))
    statements = self.transform(next(children))
    if isinstance(statements, list):
        statement_block = CodeBlock(*statements)
    else:
        statement_block = CodeBlock(statements)
    return While(condition, statement_block)