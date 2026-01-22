import os
from typing import Optional, TYPE_CHECKING
from ray.rllib.utils.annotations import PublicAPI
@property
@PublicAPI
def output_config(self):
    return self.config.get('output_config', {})