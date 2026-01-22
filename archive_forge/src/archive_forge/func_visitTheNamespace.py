from antlr4 import *
def visitTheNamespace(self, ctx: fugue_sqlParser.TheNamespaceContext):
    return self.visitChildren(ctx)