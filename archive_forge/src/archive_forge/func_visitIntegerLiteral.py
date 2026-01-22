from antlr4 import *
def visitIntegerLiteral(self, ctx: fugue_sqlParser.IntegerLiteralContext):
    return self.visitChildren(ctx)