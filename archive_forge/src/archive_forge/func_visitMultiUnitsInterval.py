from antlr4 import *
def visitMultiUnitsInterval(self, ctx: fugue_sqlParser.MultiUnitsIntervalContext):
    return self.visitChildren(ctx)