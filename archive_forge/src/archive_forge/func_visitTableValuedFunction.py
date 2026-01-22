from antlr4 import *
def visitTableValuedFunction(self, ctx: fugue_sqlParser.TableValuedFunctionContext):
    return self.visitChildren(ctx)