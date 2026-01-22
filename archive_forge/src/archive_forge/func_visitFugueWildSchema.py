from antlr4 import *
def visitFugueWildSchema(self, ctx: fugue_sqlParser.FugueWildSchemaContext):
    return self.visitChildren(ctx)