from antlr4 import *
def visitFugueJsonValue(self, ctx: fugue_sqlParser.FugueJsonValueContext):
    return self.visitChildren(ctx)