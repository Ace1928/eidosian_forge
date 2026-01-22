import functools
import weakref
import torch.nn
from torch.nn import Module
from .utils import ExactWeakKeyDictionary, is_lazy_module
def on_mutation(self, name):
    self.mutation_count += 1
    tmp = self.watchers
    self.watchers = []
    for ref in tmp:
        guarded = ref()
        if guarded is not None:
            guarded.invalidate(ref)