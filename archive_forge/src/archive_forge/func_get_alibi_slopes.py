import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank
def get_alibi_slopes(nheads):

    def get_slopes_power_of_2(nheads):
        start = 2 ** (-2 ** (-(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(nheads)]
    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return get_slopes_power_of_2(closest_power_of_2) + get_alibi_slopes(2 * closest_power_of_2)[0::2][:nheads - closest_power_of_2]