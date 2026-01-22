from antlr4 import *
def visitNullLiteral(self, ctx: fugue_sqlParser.NullLiteralContext):
    return self.visitChildren(ctx)