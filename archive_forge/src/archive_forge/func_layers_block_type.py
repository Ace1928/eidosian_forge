import math
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def layers_block_type(self):
    return ['attention' if i % self.attn_layer_period == self.attn_layer_offset else 'mamba' for i in range(self.num_hidden_layers)]