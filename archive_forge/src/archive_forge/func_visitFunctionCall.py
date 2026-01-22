from antlr4 import *
def visitFunctionCall(self, ctx: fugue_sqlParser.FunctionCallContext):
    return self.visitChildren(ctx)