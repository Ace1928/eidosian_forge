from antlr4 import *
def visitCurrentDatetime(self, ctx: fugue_sqlParser.CurrentDatetimeContext):
    return self.visitChildren(ctx)