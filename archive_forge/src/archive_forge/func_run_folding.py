import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
def run_folding(self):
    if self.const_subgraph_module is None or self.fx_const_folded_attrs_name is None:
        return
    assert not self.has_folding_been_run
    self.has_folding_been_run = True
    folded_attrs = self.const_subgraph_module()

    def _create_param(i):
        return torch.nn.Parameter(i if not isinstance(i, int) else torch.Tensor([i]).to(device=self.device_for_folded_attrs), requires_grad=i.requires_grad if isinstance(i, torch.Tensor) else False)
    params = torch.nn.ParameterList([_create_param(i) for i in folded_attrs]) if isinstance(folded_attrs, tuple) else _create_param(folded_attrs)
    setattr(self, self.fx_const_folded_attrs_name, params)