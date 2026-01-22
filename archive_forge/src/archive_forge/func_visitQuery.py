from antlr4 import *
def visitQuery(self, ctx: fugue_sqlParser.QueryContext):
    return self.visitChildren(ctx)