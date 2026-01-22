from antlr4 import *
def visitHiveChangeColumn(self, ctx: fugue_sqlParser.HiveChangeColumnContext):
    return self.visitChildren(ctx)