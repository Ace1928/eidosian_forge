from antlr4 import *
def visitAssignmentList(self, ctx: fugue_sqlParser.AssignmentListContext):
    return self.visitChildren(ctx)