from antlr4 import *
def visitQualifiedName(self, ctx: fugue_sqlParser.QualifiedNameContext):
    return self.visitChildren(ctx)