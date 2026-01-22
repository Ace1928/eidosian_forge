from antlr4 import *
def visitPivotClause(self, ctx: fugue_sqlParser.PivotClauseContext):
    return self.visitChildren(ctx)