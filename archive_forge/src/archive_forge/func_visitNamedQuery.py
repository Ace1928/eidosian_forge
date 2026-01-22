from antlr4 import *
def visitNamedQuery(self, ctx: fugue_sqlParser.NamedQueryContext):
    return self.visitChildren(ctx)