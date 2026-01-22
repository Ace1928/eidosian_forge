from antlr4 import *
def visitLogicalBinary(self, ctx: fugue_sqlParser.LogicalBinaryContext):
    return self.visitChildren(ctx)