from antlr4 import *
def visitNamespace(self, ctx: sqlParser.NamespaceContext):
    return self.visitChildren(ctx)