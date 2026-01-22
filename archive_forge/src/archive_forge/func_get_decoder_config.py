from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
def get_decoder_config(self, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, feature: str='default') -> OnnxConfig:
    """
        Returns ONNX decoder config for `VisionEncoderDecoder` model.

        Args:
            encoder_config (`PretrainedConfig`):
                The encoder model's configuration to use when exporting to ONNX.
            decoder_config (`PretrainedConfig`):
                The decoder model's configuration to use when exporting to ONNX
            feature (`str`, *optional*):
                The type of feature to export the model with.

        Returns:
            [`VisionEncoderDecoderDecoderOnnxConfig`]: An instance of the ONNX configuration object.
        """
    decoder_config.encoder_hidden_size = encoder_config.hidden_size
    return VisionEncoderDecoderDecoderOnnxConfig(decoder_config, feature)