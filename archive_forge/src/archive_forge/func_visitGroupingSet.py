from antlr4 import *
def visitGroupingSet(self, ctx: fugue_sqlParser.GroupingSetContext):
    return self.visitChildren(ctx)