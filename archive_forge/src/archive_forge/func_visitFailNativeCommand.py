from antlr4 import *
def visitFailNativeCommand(self, ctx: fugue_sqlParser.FailNativeCommandContext):
    return self.visitChildren(ctx)