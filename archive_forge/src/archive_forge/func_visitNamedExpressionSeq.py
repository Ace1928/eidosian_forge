from antlr4 import *
def visitNamedExpressionSeq(self, ctx: fugue_sqlParser.NamedExpressionSeqContext):
    return self.visitChildren(ctx)