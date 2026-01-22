from antlr4 import *
def visitFugueWildSchemaPair(self, ctx: fugue_sqlParser.FugueWildSchemaPairContext):
    return self.visitChildren(ctx)