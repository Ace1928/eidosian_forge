from antlr4 import *
def visitParenthesizedExpression(self, ctx: fugue_sqlParser.ParenthesizedExpressionContext):
    return self.visitChildren(ctx)