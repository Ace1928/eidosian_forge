from antlr4 import *
def visitFugueJsonString(self, ctx: fugue_sqlParser.FugueJsonStringContext):
    return self.visitChildren(ctx)