from antlr4 import *
def visitPredicate(self, ctx: fugue_sqlParser.PredicateContext):
    return self.visitChildren(ctx)