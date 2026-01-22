from antlr4 import *
def visitShowCreateTable(self, ctx: fugue_sqlParser.ShowCreateTableContext):
    return self.visitChildren(ctx)