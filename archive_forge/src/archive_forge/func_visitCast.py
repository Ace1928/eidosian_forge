from antlr4 import *
def visitCast(self, ctx: fugue_sqlParser.CastContext):
    return self.visitChildren(ctx)