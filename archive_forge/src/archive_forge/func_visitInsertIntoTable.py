from antlr4 import *
def visitInsertIntoTable(self, ctx: fugue_sqlParser.InsertIntoTableContext):
    return self.visitChildren(ctx)