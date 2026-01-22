from antlr4 import *
def visitSetOperation(self, ctx: fugue_sqlParser.SetOperationContext):
    return self.visitChildren(ctx)