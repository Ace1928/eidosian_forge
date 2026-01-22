from antlr4 import *
def visitCtes(self, ctx: fugue_sqlParser.CtesContext):
    return self.visitChildren(ctx)