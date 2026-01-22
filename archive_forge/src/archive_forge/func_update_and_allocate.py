import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
def update_and_allocate(self, model, global_step, force_mask=False):
    if global_step < self.peft_config.total_step - self.peft_config.tfinal:
        self.update_ipt(model)
    budget, mask_ind = self.budget_schedule(global_step)
    if mask_ind or force_mask:
        rank_pattern = self.mask_to_budget(model, budget)
    else:
        rank_pattern = None
    return (budget, rank_pattern)