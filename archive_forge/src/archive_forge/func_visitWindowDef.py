from antlr4 import *
def visitWindowDef(self, ctx: fugue_sqlParser.WindowDefContext):
    return self.visitChildren(ctx)