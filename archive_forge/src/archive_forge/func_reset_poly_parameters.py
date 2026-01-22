import math
from typing import Any
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer
from .config import PolyConfig
from .router import get_router
def reset_poly_parameters(self, adapter_name, init_weights):
    if adapter_name in self.poly_lora_A.keys():
        n_splits, n_skills, d, r = self.poly_lora_A[adapter_name].shape
        for skill in range(n_skills):
            for split in range(n_splits):
                param = torch.empty((r, d))
                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                self.poly_lora_A[adapter_name].data[split, skill, :, :] = param.T
        if init_weights:
            torch.nn.init.zeros_(self.poly_lora_B[adapter_name])
        else:
            n_splits, n_skills, r, d = self.poly_lora_B[adapter_name].shape
            for skill in range(n_skills):
                for split in range(n_splits):
                    param = torch.empty((d, r))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.poly_lora_B[adapter_name].data[split, skill, :, :] = param.T
        self.poly_router[adapter_name].reset()