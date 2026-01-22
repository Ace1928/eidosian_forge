from antlr4 import *
def visitStar(self, ctx: fugue_sqlParser.StarContext):
    return self.visitChildren(ctx)