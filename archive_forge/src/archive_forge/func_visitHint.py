from antlr4 import *
def visitHint(self, ctx: fugue_sqlParser.HintContext):
    return self.visitChildren(ctx)