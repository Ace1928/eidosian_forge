from antlr4 import *
def visitTrim(self, ctx: fugue_sqlParser.TrimContext):
    return self.visitChildren(ctx)