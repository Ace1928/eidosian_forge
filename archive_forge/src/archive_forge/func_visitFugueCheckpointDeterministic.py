from antlr4 import *
def visitFugueCheckpointDeterministic(self, ctx: fugue_sqlParser.FugueCheckpointDeterministicContext):
    return self.visitChildren(ctx)