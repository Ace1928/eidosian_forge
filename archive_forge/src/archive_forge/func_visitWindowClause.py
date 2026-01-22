from antlr4 import *
def visitWindowClause(self, ctx: fugue_sqlParser.WindowClauseContext):
    return self.visitChildren(ctx)