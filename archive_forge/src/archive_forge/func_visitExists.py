from antlr4 import *
def visitExists(self, ctx: fugue_sqlParser.ExistsContext):
    return self.visitChildren(ctx)