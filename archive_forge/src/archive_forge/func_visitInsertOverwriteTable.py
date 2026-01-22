from antlr4 import *
def visitInsertOverwriteTable(self, ctx: fugue_sqlParser.InsertOverwriteTableContext):
    return self.visitChildren(ctx)