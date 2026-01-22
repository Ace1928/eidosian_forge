from antlr4 import *
def visitFugueDataFramesList(self, ctx: fugue_sqlParser.FugueDataFramesListContext):
    return self.visitChildren(ctx)