from antlr4 import *
def visitFugueRenameColumnsTask(self, ctx: fugue_sqlParser.FugueRenameColumnsTaskContext):
    return self.visitChildren(ctx)