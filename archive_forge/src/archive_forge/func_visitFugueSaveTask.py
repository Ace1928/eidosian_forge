from antlr4 import *
def visitFugueSaveTask(self, ctx: fugue_sqlParser.FugueSaveTaskContext):
    return self.visitChildren(ctx)