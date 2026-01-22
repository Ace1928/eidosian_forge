from antlr4 import *
def visitResource(self, ctx: fugue_sqlParser.ResourceContext):
    return self.visitChildren(ctx)