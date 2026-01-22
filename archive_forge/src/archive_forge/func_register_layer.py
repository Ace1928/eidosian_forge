from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def register_layer(self, layer: nn.Module, aggregate_fn=None, reduce_fn=None, mask_fn=None, features=None, feature_dim=None, **sparse_config):
    """
        Registers a layer for sparsification. The layer should be part of self.model.
        Specifically, registers a pre-forward hook to the layer. The hook will apply the aggregate_fn
        and store the aggregated activations that is input over each step.

        Note::
            - There is no need to pass in the name of the layer as it is automatically computed as per
              the fqn convention.

            - All the functions (fn) passed as argument will be called at a dim, feature level.
        """
    name = module_to_fqn(self.model, layer)
    assert name is not None, 'layer not found in the model'
    if name in self.data_groups:
        warnings.warn('layer already attached to the sparsifier, deregistering the layer and registering with new config')
        self.unregister_layer(name=name)
    local_args = copy.deepcopy(self.defaults)
    update_dict = {'aggregate_fn': aggregate_fn, 'reduce_fn': reduce_fn, 'mask_fn': mask_fn, 'features': features, 'feature_dim': feature_dim, 'layer': layer}
    local_args.update(((arg, val) for arg, val in update_dict.items() if val is not None))
    local_args['sparse_config'].update(sparse_config)
    self._safe_rail_checks(local_args)
    self.data_groups[name] = local_args
    agg_hook = layer.register_forward_pre_hook(self._aggregate_hook(name=name))
    self.state[name]['mask'] = None
    self.data_groups[name]['hook'] = agg_hook
    self.data_groups[name]['hook_state'] = 'aggregate'