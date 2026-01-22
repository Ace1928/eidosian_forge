from antlr4 import *
def visitStorageHandler(self, ctx: fugue_sqlParser.StorageHandlerContext):
    return self.visitChildren(ctx)