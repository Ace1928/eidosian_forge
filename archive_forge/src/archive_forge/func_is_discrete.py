import torch
@property
def is_discrete(self):
    return any((c.is_discrete for c in self.cseq))