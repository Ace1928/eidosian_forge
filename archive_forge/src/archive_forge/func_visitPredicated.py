from antlr4 import *
def visitPredicated(self, ctx: fugue_sqlParser.PredicatedContext):
    return self.visitChildren(ctx)