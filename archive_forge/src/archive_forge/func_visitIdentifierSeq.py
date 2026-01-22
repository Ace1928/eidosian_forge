from antlr4 import *
def visitIdentifierSeq(self, ctx: fugue_sqlParser.IdentifierSeqContext):
    return self.visitChildren(ctx)