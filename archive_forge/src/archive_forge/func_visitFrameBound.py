from antlr4 import *
def visitFrameBound(self, ctx: fugue_sqlParser.FrameBoundContext):
    return self.visitChildren(ctx)