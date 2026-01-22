from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
@classmethod
def from_encoder_decoder_configs(cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
    """
        Instantiate a [`VisionEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`VisionEncoderDecoderConfig`]: An instance of a configuration object
        """
    logger.info('Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config')
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)