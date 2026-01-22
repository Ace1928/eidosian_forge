from antlr4 import *
def visitFugueExtension(self, ctx: fugue_sqlParser.FugueExtensionContext):
    return self.visitChildren(ctx)