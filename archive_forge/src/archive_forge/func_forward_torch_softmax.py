import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
def forward_torch_softmax(self, input, mask):
    if self.input_in_float16 and self.softmax_in_fp32:
        input = input.float()
    if self.scale is not None:
        input = input * self.scale
    mask_output = self.mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    if self.input_in_float16 and self.softmax_in_fp32:
        if self.input_in_fp16:
            probs = probs.half()
        else:
            probs = probs.bfloat16()
    return probs