from antlr4 import *
def visitShowColumns(self, ctx: fugue_sqlParser.ShowColumnsContext):
    return self.visitChildren(ctx)