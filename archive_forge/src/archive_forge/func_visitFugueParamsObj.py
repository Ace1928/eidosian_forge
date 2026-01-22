from antlr4 import *
def visitFugueParamsObj(self, ctx: fugue_sqlParser.FugueParamsObjContext):
    return self.visitChildren(ctx)