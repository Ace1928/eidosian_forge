from antlr4 import *
def visitFugueYield(self, ctx: fugue_sqlParser.FugueYieldContext):
    return self.visitChildren(ctx)