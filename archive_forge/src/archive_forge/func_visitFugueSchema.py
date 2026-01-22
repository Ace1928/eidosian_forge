from antlr4 import *
def visitFugueSchema(self, ctx: fugue_sqlParser.FugueSchemaContext):
    return self.visitChildren(ctx)