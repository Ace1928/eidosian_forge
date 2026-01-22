from antlr4 import *
def visitFugueSingleStatement(self, ctx: fugue_sqlParser.FugueSingleStatementContext):
    return self.visitChildren(ctx)