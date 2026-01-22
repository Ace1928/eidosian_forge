from antlr4 import *
def visitSetNamespaceLocation(self, ctx: fugue_sqlParser.SetNamespaceLocationContext):
    return self.visitChildren(ctx)