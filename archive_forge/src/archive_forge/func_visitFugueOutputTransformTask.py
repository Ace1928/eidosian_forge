from antlr4 import *
def visitFugueOutputTransformTask(self, ctx: fugue_sqlParser.FugueOutputTransformTaskContext):
    return self.visitChildren(ctx)