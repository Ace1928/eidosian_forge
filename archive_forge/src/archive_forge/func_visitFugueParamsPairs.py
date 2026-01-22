from antlr4 import *
def visitFugueParamsPairs(self, ctx: fugue_sqlParser.FugueParamsPairsContext):
    return self.visitChildren(ctx)