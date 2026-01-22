from antlr4 import *
def visitLoadData(self, ctx: fugue_sqlParser.LoadDataContext):
    return self.visitChildren(ctx)