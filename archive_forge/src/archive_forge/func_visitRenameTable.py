from antlr4 import *
def visitRenameTable(self, ctx: fugue_sqlParser.RenameTableContext):
    return self.visitChildren(ctx)