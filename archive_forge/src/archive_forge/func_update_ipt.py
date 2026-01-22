import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
def update_ipt(self, model):
    for n, p in model.named_parameters():
        if 'lora_' in n and self.adapter_name in n:
            if n not in self.ipt:
                self.ipt[n] = torch.zeros_like(p)
                self.exp_avg_ipt[n] = torch.zeros_like(p)
                self.exp_avg_unc[n] = torch.zeros_like(p)
            with torch.no_grad():
                self.ipt[n] = (p * p.grad).abs().detach()
                self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()