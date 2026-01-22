from antlr4 import *
def visitFugueSampleTask(self, ctx: fugue_sqlParser.FugueSampleTaskContext):
    return self.visitChildren(ctx)