from antlr4 import *
def visitSkewSpec(self, ctx: fugue_sqlParser.SkewSpecContext):
    return self.visitChildren(ctx)