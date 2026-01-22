from antlr4 import *
def visitFugueSchemaStructType(self, ctx: fugue_sqlParser.FugueSchemaStructTypeContext):
    return self.visitChildren(ctx)