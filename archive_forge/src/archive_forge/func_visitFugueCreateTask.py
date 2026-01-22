from antlr4 import *
def visitFugueCreateTask(self, ctx: fugue_sqlParser.FugueCreateTaskContext):
    return self.visitChildren(ctx)