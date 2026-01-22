from antlr4 import *
def visitCreateTableLike(self, ctx: fugue_sqlParser.CreateTableLikeContext):
    return self.visitChildren(ctx)