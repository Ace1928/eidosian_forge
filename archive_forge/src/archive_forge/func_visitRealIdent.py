from antlr4 import *
def visitRealIdent(self, ctx: sqlParser.RealIdentContext):
    return self.visitChildren(ctx)