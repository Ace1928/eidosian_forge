from antlr4 import *
def visitRowFormatSerde(self, ctx: fugue_sqlParser.RowFormatSerdeContext):
    return self.visitChildren(ctx)