from antlr4 import *
def visitFugueSingleFile(self, ctx: fugue_sqlParser.FugueSingleFileContext):
    return self.visitChildren(ctx)