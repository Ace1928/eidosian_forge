from antlr4 import *
def visitSearchedCase(self, ctx: fugue_sqlParser.SearchedCaseContext):
    return self.visitChildren(ctx)