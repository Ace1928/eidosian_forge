import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING
@classmethod
def from_vision_qformer_text_configs(cls, vision_config: InstructBlipVisionConfig, qformer_config: InstructBlipQFormerConfig, text_config: PretrainedConfig, **kwargs):
    """
        Instantiate a [`InstructBlipConfig`] (or a derived class) from a InstructBLIP vision model, Q-Former and
        language model configurations.

        Returns:
            [`InstructBlipConfig`]: An instance of a configuration object
        """
    return cls(vision_config=vision_config.to_dict(), qformer_config=qformer_config.to_dict(), text_config=text_config.to_dict(), **kwargs)