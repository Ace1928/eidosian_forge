from antlr4 import *
def visitFugueJsonObj(self, ctx: fugue_sqlParser.FugueJsonObjContext):
    return self.visitChildren(ctx)