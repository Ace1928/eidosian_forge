from antlr4 import *
def visitFugueLoadColumns(self, ctx: fugue_sqlParser.FugueLoadColumnsContext):
    return self.visitChildren(ctx)