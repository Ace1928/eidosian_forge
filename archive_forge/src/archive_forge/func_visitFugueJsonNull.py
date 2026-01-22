from antlr4 import *
def visitFugueJsonNull(self, ctx: fugue_sqlParser.FugueJsonNullContext):
    return self.visitChildren(ctx)