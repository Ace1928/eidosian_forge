from antlr4 import *
def visitNotMatchedClause(self, ctx: fugue_sqlParser.NotMatchedClauseContext):
    return self.visitChildren(ctx)