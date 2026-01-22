from antlr4 import *
def visitFugueSaveMode(self, ctx: fugue_sqlParser.FugueSaveModeContext):
    return self.visitChildren(ctx)