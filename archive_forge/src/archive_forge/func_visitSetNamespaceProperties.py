from antlr4 import *
def visitSetNamespaceProperties(self, ctx: fugue_sqlParser.SetNamespacePropertiesContext):
    return self.visitChildren(ctx)