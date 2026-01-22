from antlr4 import *
def visitQuotedIdentifierAlternative(self, ctx: fugue_sqlParser.QuotedIdentifierAlternativeContext):
    return self.visitChildren(ctx)