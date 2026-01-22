from antlr4 import *
def visitQualifiedColTypeWithPosition(self, ctx: fugue_sqlParser.QualifiedColTypeWithPositionContext):
    return self.visitChildren(ctx)