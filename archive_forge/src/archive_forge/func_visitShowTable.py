from antlr4 import *
def visitShowTable(self, ctx: fugue_sqlParser.ShowTableContext):
    return self.visitChildren(ctx)