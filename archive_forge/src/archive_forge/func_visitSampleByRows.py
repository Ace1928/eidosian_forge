from antlr4 import *
def visitSampleByRows(self, ctx: fugue_sqlParser.SampleByRowsContext):
    return self.visitChildren(ctx)