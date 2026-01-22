from antlr4 import *
def visitFugueDataFrameSource(self, ctx: fugue_sqlParser.FugueDataFrameSourceContext):
    return self.visitChildren(ctx)