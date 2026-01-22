from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def squash_mask(self, attach_sparsify_hook=True, **kwargs):
    """
        Unregisters aggregate hook that was applied earlier and registers sparsification hooks if
        attach_sparsify_hook = True.
        """
    for name, configs in self.data_groups.items():
        configs['hook'].remove()
        configs.pop('hook')
        self.data_groups[name]['hook_state'] = 'None'
        if attach_sparsify_hook:
            configs['hook'] = configs['layer'].register_forward_pre_hook(self._sparsify_hook(name))
        configs['hook_state'] = 'sparsify'