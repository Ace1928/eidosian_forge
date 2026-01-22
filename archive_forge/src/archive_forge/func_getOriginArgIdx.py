from __future__ import annotations
from ..runtime import driver
def getOriginArgIdx(self, idx, args):
    if self.ids_of_folded_args:
        ids_before_folding_arg = [i for i in range(len(args)) if i not in self.ids_of_folded_args]
        return ids_before_folding_arg[idx]
    else:
        return idx