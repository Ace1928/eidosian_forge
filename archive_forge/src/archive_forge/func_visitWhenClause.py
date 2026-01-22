from antlr4 import *
def visitWhenClause(self, ctx: fugue_sqlParser.WhenClauseContext):
    return self.visitChildren(ctx)