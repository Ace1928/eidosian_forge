from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..chinese_clip.configuration_chinese_clip import ChineseCLIPVisionConfig
from ..clip.configuration_clip import CLIPVisionConfig
from ..siglip.configuration_siglip import SiglipVisionConfig
@classmethod
def from_vision_text_configs(cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs):
    """
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """
    return cls(vision_config=vision_config.to_dict(), text_config=text_config.to_dict(), **kwargs)