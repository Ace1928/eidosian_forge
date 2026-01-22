from antlr4 import *
def visitCommentTable(self, ctx: fugue_sqlParser.CommentTableContext):
    return self.visitChildren(ctx)