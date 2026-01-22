from antlr4 import *
def visitFuguePartitionNum(self, ctx: fugue_sqlParser.FuguePartitionNumContext):
    return self.visitChildren(ctx)