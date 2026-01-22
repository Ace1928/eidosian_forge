from antlr4 import *
def visitColTypeList(self, ctx: fugue_sqlParser.ColTypeListContext):
    return self.visitChildren(ctx)