from antlr4 import *
def visitCreateFunction(self, ctx: fugue_sqlParser.CreateFunctionContext):
    return self.visitChildren(ctx)