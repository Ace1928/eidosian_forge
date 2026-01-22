from antlr4 import *
def visitFugueJsonKey(self, ctx: fugue_sqlParser.FugueJsonKeyContext):
    return self.visitChildren(ctx)