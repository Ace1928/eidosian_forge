from antlr4 import *
def visitSingleInsertQuery(self, ctx: fugue_sqlParser.SingleInsertQueryContext):
    return self.visitChildren(ctx)