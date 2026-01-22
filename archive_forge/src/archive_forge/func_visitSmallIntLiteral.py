from antlr4 import *
def visitSmallIntLiteral(self, ctx: fugue_sqlParser.SmallIntLiteralContext):
    return self.visitChildren(ctx)