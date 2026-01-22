from antlr4 import *
def visitShowFunctions(self, ctx: fugue_sqlParser.ShowFunctionsContext):
    return self.visitChildren(ctx)