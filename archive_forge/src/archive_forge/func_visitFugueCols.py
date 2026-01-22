from antlr4 import *
def visitFugueCols(self, ctx: fugue_sqlParser.FugueColsContext):
    return self.visitChildren(ctx)