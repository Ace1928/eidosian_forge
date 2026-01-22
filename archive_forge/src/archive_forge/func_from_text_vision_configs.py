import os
from typing import TYPE_CHECKING, List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@classmethod
def from_text_vision_configs(cls, text_config: AlignTextConfig, vision_config: AlignVisionConfig, **kwargs):
    """
        Instantiate a [`AlignConfig`] (or a derived class) from align text model configuration and align vision model
        configuration.

        Returns:
            [`AlignConfig`]: An instance of a configuration object
        """
    return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)