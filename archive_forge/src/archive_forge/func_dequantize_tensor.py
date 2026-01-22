import logging
from typing import Union
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
def dequantize_tensor(self, qweight, max_abs):
    qweight_flatten = qweight.flatten()
    weight_normed = self.norm_lookup_table[qweight_flatten]
    weight = weight_normed * max_abs
    weight = weight.reshape(qweight.shape)
    return weight