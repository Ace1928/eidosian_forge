from antlr4 import *
def visitBigDecimalLiteral(self, ctx: fugue_sqlParser.BigDecimalLiteralContext):
    return self.visitChildren(ctx)