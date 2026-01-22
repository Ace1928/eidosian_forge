from antlr4 import *
def visitFugueSqlEngine(self, ctx: fugue_sqlParser.FugueSqlEngineContext):
    return self.visitChildren(ctx)