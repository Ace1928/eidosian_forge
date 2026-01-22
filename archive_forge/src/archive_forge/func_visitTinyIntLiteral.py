from antlr4 import *
def visitTinyIntLiteral(self, ctx: fugue_sqlParser.TinyIntLiteralContext):
    return self.visitChildren(ctx)