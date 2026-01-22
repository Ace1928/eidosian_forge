from antlr4 import *
def visitArithmeticOperator(self, ctx: fugue_sqlParser.ArithmeticOperatorContext):
    return self.visitChildren(ctx)