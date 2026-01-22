from antlr4 import *
def visitDescribeFuncName(self, ctx: fugue_sqlParser.DescribeFuncNameContext):
    return self.visitChildren(ctx)