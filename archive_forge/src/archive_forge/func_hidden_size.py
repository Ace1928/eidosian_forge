from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def hidden_size(self) -> int:
    return self.d_model