from antlr4 import *
def visitFugueSchemaKey(self, ctx: fugue_sqlParser.FugueSchemaKeyContext):
    return self.visitChildren(ctx)