from antlr4 import *
def visitMultipartIdentifier(self, ctx: fugue_sqlParser.MultipartIdentifierContext):
    return self.visitChildren(ctx)