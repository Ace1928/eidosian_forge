from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
@classmethod
def from_sub_models_config(cls, text_encoder_config: PretrainedConfig, audio_encoder_config: PretrainedConfig, decoder_config: MusicgenDecoderConfig, **kwargs):
    """
        Instantiate a [`MusicgenConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenConfig`]: An instance of a configuration object
        """
    return cls(text_encoder=text_encoder_config.to_dict(), audio_encoder=audio_encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)