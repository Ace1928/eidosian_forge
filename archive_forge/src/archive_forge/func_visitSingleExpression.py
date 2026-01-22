from antlr4 import *
def visitSingleExpression(self, ctx: fugue_sqlParser.SingleExpressionContext):
    return self.visitChildren(ctx)