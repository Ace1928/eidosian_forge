from antlr4 import *
def visitApplyTransform(self, ctx: fugue_sqlParser.ApplyTransformContext):
    return self.visitChildren(ctx)