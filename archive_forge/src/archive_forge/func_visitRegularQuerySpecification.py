from antlr4 import *
def visitRegularQuerySpecification(self, ctx: fugue_sqlParser.RegularQuerySpecificationContext):
    return self.visitChildren(ctx)