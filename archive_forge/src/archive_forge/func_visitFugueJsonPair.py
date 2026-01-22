from antlr4 import *
def visitFugueJsonPair(self, ctx: fugue_sqlParser.FugueJsonPairContext):
    return self.visitChildren(ctx)