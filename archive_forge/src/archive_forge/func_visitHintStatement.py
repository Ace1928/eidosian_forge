from antlr4 import *
def visitHintStatement(self, ctx: fugue_sqlParser.HintStatementContext):
    return self.visitChildren(ctx)