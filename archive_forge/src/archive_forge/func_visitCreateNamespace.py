from antlr4 import *
def visitCreateNamespace(self, ctx: fugue_sqlParser.CreateNamespaceContext):
    return self.visitChildren(ctx)