from antlr4 import *
def visitFugueRenamePair(self, ctx: fugue_sqlParser.FugueRenamePairContext):
    return self.visitChildren(ctx)