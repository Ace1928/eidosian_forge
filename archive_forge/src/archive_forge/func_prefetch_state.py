from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def prefetch_state(self, p):
    if self.is_paged:
        state = self.state[p]
        s1 = state['state1']
        is_paged = getattr(s1, 'is_paged', False)
        if is_paged:
            F.prefetch_tensor(state['state1'])
            if 'state2' in state:
                F.prefetch_tensor(state['state2'])