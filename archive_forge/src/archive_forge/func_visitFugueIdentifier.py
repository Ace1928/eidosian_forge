from antlr4 import *
def visitFugueIdentifier(self, ctx: fugue_sqlParser.FugueIdentifierContext):
    return self.visitChildren(ctx)