from sympy.external import import_module
def visit_Sequence(self, seq):
    """Visitor Function for code sequence

            Visits a code sequence/ block and calls the visitor function on all the
            children of the code block to create corresponding code in python

            """
    if seq is not None:
        for node in seq:
            self._py_ast.append(call_visitor(node))