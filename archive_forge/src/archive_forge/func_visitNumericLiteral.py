from antlr4 import *
def visitNumericLiteral(self, ctx: fugue_sqlParser.NumericLiteralContext):
    return self.visitChildren(ctx)