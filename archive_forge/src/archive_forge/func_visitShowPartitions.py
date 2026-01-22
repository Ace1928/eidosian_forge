from antlr4 import *
def visitShowPartitions(self, ctx: fugue_sqlParser.ShowPartitionsContext):
    return self.visitChildren(ctx)