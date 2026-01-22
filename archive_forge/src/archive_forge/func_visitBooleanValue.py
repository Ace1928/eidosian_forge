from antlr4 import *
def visitBooleanValue(self, ctx: fugue_sqlParser.BooleanValueContext):
    return self.visitChildren(ctx)