from antlr4 import *
def visitAlterColumnAction(self, ctx: fugue_sqlParser.AlterColumnActionContext):
    return self.visitChildren(ctx)