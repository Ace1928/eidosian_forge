from antlr4 import *
def visitFugueTransformTask(self, ctx: fugue_sqlParser.FugueTransformTaskContext):
    return self.visitChildren(ctx)