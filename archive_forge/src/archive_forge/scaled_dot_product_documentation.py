import logging
from dataclasses import dataclass
from typing import Optional, Union
import torch
from torch import nn
from xformers.components.attention import (
from xformers.components.attention.core import scaled_dot_product_attention

        att_mask    A 2D or 3D mask which ignores attention at certain positions.

                    - If the mask is boolean, a value of True will keep the value,
                        while a value of False will mask the value.

                        Key padding masks (dimension: batch x sequence length) and attention masks
                        (dimension: sequence length x sequence length OR batch x sequence length x sequence length)
                        can be combined and passed in here. Method maybe_merge_masks provided in the utils can be
                        used for that merging.

                    - If the mask has the float type, then an additive mask is expected (masked values are -inf)

        