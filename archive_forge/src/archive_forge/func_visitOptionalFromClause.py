from antlr4 import *
def visitOptionalFromClause(self, ctx: fugue_sqlParser.OptionalFromClauseContext):
    return self.visitChildren(ctx)