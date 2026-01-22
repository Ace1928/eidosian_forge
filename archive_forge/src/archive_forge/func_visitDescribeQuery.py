from antlr4 import *
def visitDescribeQuery(self, ctx: fugue_sqlParser.DescribeQueryContext):
    return self.visitChildren(ctx)