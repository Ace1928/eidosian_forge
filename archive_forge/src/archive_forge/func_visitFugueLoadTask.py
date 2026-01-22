from antlr4 import *
def visitFugueLoadTask(self, ctx: fugue_sqlParser.FugueLoadTaskContext):
    return self.visitChildren(ctx)