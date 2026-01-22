from antlr4 import *
def visitDoubleLiteral(self, ctx: fugue_sqlParser.DoubleLiteralContext):
    return self.visitChildren(ctx)