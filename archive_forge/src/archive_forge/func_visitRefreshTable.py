from antlr4 import *
def visitRefreshTable(self, ctx: fugue_sqlParser.RefreshTableContext):
    return self.visitChildren(ctx)