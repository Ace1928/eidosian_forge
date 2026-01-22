from antlr4 import *
def visitManageResource(self, ctx: fugue_sqlParser.ManageResourceContext):
    return self.visitChildren(ctx)