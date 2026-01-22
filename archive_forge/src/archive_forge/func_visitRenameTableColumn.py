from antlr4 import *
def visitRenameTableColumn(self, ctx: fugue_sqlParser.RenameTableColumnContext):
    return self.visitChildren(ctx)