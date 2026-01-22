from antlr4 import *
def visitFugueZipType(self, ctx: fugue_sqlParser.FugueZipTypeContext):
    return self.visitChildren(ctx)