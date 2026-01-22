from antlr4 import *
def visitAliasedRelation(self, ctx: fugue_sqlParser.AliasedRelationContext):
    return self.visitChildren(ctx)