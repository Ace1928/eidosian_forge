from antlr4 import *
def visitCacheTable(self, ctx: fugue_sqlParser.CacheTableContext):
    return self.visitChildren(ctx)