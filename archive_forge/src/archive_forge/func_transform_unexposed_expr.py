from sympy.external import import_module
import os
def transform_unexposed_expr(self, node):
    """Transformation function for unexposed expression

            Unexposed expressions are used to wrap float, double literals and
            expressions

            Returns
            =======

            expr : Codegen AST Node
                the result from the wrapped expression

            None : NoneType
                No childs are found for the node

            Raises
            ======

            ValueError if the expression contains multiple children

            """
    try:
        children = node.get_children()
        expr = self.transform(next(children))
    except StopIteration:
        return None
    try:
        next(children)
        raise ValueError('Unexposed expression has > 1 children.')
    except StopIteration:
        pass
    return expr