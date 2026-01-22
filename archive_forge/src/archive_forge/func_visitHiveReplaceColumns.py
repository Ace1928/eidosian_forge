from antlr4 import *
def visitHiveReplaceColumns(self, ctx: fugue_sqlParser.HiveReplaceColumnsContext):
    return self.visitChildren(ctx)