from antlr4 import *
def visitFugueProcessTask(self, ctx: fugue_sqlParser.FugueProcessTaskContext):
    return self.visitChildren(ctx)