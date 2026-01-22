from antlr4 import *
def visitUncacheTable(self, ctx: fugue_sqlParser.UncacheTableContext):
    return self.visitChildren(ctx)