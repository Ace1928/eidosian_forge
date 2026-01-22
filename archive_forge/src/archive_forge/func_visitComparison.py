from antlr4 import *
def visitComparison(self, ctx: fugue_sqlParser.ComparisonContext):
    return self.visitChildren(ctx)