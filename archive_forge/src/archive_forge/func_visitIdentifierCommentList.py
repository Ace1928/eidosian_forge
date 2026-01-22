from antlr4 import *
def visitIdentifierCommentList(self, ctx: fugue_sqlParser.IdentifierCommentListContext):
    return self.visitChildren(ctx)