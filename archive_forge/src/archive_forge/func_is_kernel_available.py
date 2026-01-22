import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
def is_kernel_available(self, mask, b, np, sq, sk):
    attn_batches = b * np
    if self.scaled_masked_softmax_fusion and self.input_in_float16 and (self.attn_mask_type == AttnMaskType.causal or (self.attn_mask_type == AttnMaskType.padding and mask is not None)) and (16 < sk <= 8192) and (sq % 4 == 0) and (sk % 4 == 0) and (attn_batches % 4 == 0):
        if 0 <= sk <= 8192:
            batch_per_block = self.get_batch_per_block(sq, sk, b, np)
            if self.attn_mask_type == AttnMaskType.causal:
                if attn_batches % batch_per_block == 0:
                    return True
            elif sq % batch_per_block == 0:
                return True
    return False