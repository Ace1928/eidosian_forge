import math
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def layers_num_experts(self):
    return [self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1 for i in range(self.num_hidden_layers)]