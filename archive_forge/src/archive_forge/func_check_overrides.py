from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def check_overrides(self):
    for module, attr, config in self.mng.module_weight_config_triple:
        pmodule = getattr(module, attr)
        assert pmodule is not None
        assert isinstance(pmodule, torch.Tensor) or isinstance(pmodule, torch.Parameter)
        found = False
        for gindex, group in enumerate(self.param_groups):
            if found:
                break
            for pindex, p in enumerate(group['params']):
                if found:
                    break
                if id(p) == id(pmodule):
                    self.mng.pid2config[id(p)] = config
                    self.mng.index2config[gindex, pindex] = self.mng.pid2config[id(p)]
                    found = True