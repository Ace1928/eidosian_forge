from antlr4 import *
def visitSubscript(self, ctx: fugue_sqlParser.SubscriptContext):
    return self.visitChildren(ctx)