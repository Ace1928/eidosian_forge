from antlr4 import *
def visitJoinCriteria(self, ctx: fugue_sqlParser.JoinCriteriaContext):
    return self.visitChildren(ctx)