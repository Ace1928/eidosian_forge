from antlr4 import *
def visitFuguePath(self, ctx: fugue_sqlParser.FuguePathContext):
    return self.visitChildren(ctx)