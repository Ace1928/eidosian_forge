from antlr4 import *
def visitSampleByBucket(self, ctx: fugue_sqlParser.SampleByBucketContext):
    return self.visitChildren(ctx)