from antlr4 import *
def visitLateralView(self, ctx: fugue_sqlParser.LateralViewContext):
    return self.visitChildren(ctx)