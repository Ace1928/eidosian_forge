from sympy.external import import_module
def visit_TranslationUnit(self, node):
    """
            Function to visit all the elements of the Translation Unit
            created by LFortran ASR
            """
    for s in node.global_scope.symbols:
        sym = node.global_scope.symbols[s]
        self.visit(sym)
    for item in node.items:
        self.visit(item)