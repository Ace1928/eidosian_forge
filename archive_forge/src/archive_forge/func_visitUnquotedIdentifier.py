from antlr4 import *
def visitUnquotedIdentifier(self, ctx: fugue_sqlParser.UnquotedIdentifierContext):
    return self.visitChildren(ctx)