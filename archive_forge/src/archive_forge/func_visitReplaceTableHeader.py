from antlr4 import *
def visitReplaceTableHeader(self, ctx: fugue_sqlParser.ReplaceTableHeaderContext):
    return self.visitChildren(ctx)