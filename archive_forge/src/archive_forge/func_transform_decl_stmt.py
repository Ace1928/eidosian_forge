from sympy.external import import_module
import os
def transform_decl_stmt(self, node):
    """Transformation function for declaration statements

            These statements are used to wrap different kinds of declararions
            like variable or function declaration
            The function calls the transformer function for the child of the
            given node

            Returns
            =======

            statement : Codegen AST Node
                contains the node returned by the children node for the type of
                declaration

            Raises
            ======

            ValueError if multiple children present

            """
    try:
        children = node.get_children()
        statement = self.transform(next(children))
    except StopIteration:
        pass
    try:
        self.transform(next(children))
        raise ValueError("Don't know how to handle multiple statements")
    except StopIteration:
        pass
    return statement