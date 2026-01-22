from antlr4 import *
def visitQueryTermDefault(self, ctx: fugue_sqlParser.QueryTermDefaultContext):
    return self.visitChildren(ctx)