from antlr4 import *
def visitAlterViewQuery(self, ctx: fugue_sqlParser.AlterViewQueryContext):
    return self.visitChildren(ctx)