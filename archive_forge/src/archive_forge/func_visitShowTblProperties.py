from antlr4 import *
def visitShowTblProperties(self, ctx: fugue_sqlParser.ShowTblPropertiesContext):
    return self.visitChildren(ctx)