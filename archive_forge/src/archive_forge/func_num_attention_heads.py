from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def num_attention_heads(self) -> int:
    return self.encoder_attention_heads