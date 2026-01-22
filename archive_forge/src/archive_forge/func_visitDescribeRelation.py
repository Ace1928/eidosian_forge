from antlr4 import *
def visitDescribeRelation(self, ctx: fugue_sqlParser.DescribeRelationContext):
    return self.visitChildren(ctx)