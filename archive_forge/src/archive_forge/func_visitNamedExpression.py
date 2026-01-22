from antlr4 import *
def visitNamedExpression(self, ctx: fugue_sqlParser.NamedExpressionContext):
    return self.visitChildren(ctx)