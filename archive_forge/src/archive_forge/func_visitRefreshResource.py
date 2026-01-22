from antlr4 import *
def visitRefreshResource(self, ctx: fugue_sqlParser.RefreshResourceContext):
    return self.visitChildren(ctx)