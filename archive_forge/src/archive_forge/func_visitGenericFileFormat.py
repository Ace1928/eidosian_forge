from antlr4 import *
def visitGenericFileFormat(self, ctx: fugue_sqlParser.GenericFileFormatContext):
    return self.visitChildren(ctx)