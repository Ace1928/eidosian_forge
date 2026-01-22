from antlr4 import *
def visitSetTableLocation(self, ctx: fugue_sqlParser.SetTableLocationContext):
    return self.visitChildren(ctx)