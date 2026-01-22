from antlr4 import *
def visitInterval(self, ctx: fugue_sqlParser.IntervalContext):
    return self.visitChildren(ctx)