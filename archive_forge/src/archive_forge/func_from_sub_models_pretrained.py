import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
@classmethod
def from_sub_models_pretrained(cls, text_encoder_pretrained_model_name_or_path: str=None, audio_encoder_pretrained_model_name_or_path: str=None, decoder_pretrained_model_name_or_path: str=None, *model_args, **kwargs) -> PreTrainedModel:
    """
        Instantiate a text encoder, an audio encoder, and a MusicGen decoder from one, two or three base classes of the
        library from pretrained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            text_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the audio encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text encoder configuration, use the prefix *text_encoder_* for each configuration
                  parameter.
                - To update the audio encoder configuration, use the prefix *audio_encoder_* for each configuration
                  parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> # initialize a musicgen model from a t5 text encoder, encodec audio encoder, and musicgen decoder
        >>> model = MusicgenForConditionalGeneration.from_sub_models_pretrained(
        ...     text_encoder_pretrained_model_name_or_path="google-t5/t5-base",
        ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",
        ...     decoder_pretrained_model_name_or_path="facebook/musicgen-small",
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./musicgen-ft")
        >>> # load fine-tuned model
        >>> model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-ft")
        ```"""
    kwargs_text_encoder = {argument[len('text_encoder_'):]: value for argument, value in kwargs.items() if argument.startswith('text_encoder_')}
    kwargs_audio_encoder = {argument[len('audio_encoder_'):]: value for argument, value in kwargs.items() if argument.startswith('audio_encoder_')}
    kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in kwargs.items() if argument.startswith('decoder_')}
    for key in kwargs_text_encoder.keys():
        del kwargs['text_encoder_' + key]
    for key in kwargs_audio_encoder.keys():
        del kwargs['audio_encoder_' + key]
    for key in kwargs_decoder.keys():
        del kwargs['decoder_' + key]
    text_encoder = kwargs_text_encoder.pop('model', None)
    if text_encoder is None:
        if text_encoder_pretrained_model_name_or_path is None:
            raise ValueError('If `text_encoder_model` is not defined as an argument, a `text_encoder_pretrained_model_name_or_path` has to be defined.')
        if 'config' not in kwargs_text_encoder:
            encoder_config, kwargs_text_encoder = AutoConfig.from_pretrained(text_encoder_pretrained_model_name_or_path, **kwargs_text_encoder, return_unused_kwargs=True)
            if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                logger.info(f'Initializing {text_encoder_pretrained_model_name_or_path} as a text_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                encoder_config.is_decoder = False
                encoder_config.add_cross_attention = False
            kwargs_text_encoder['config'] = encoder_config
        text_encoder = AutoModel.from_pretrained(text_encoder_pretrained_model_name_or_path, *model_args, **kwargs_text_encoder)
    audio_encoder = kwargs_audio_encoder.pop('model', None)
    if audio_encoder is None:
        if audio_encoder_pretrained_model_name_or_path is None:
            raise ValueError('If `audio_encoder_model` is not defined as an argument, an `audio_encoder_pretrained_model_name_or_path` has to be defined.')
        if 'config' not in kwargs_audio_encoder:
            encoder_config, kwargs_audio_encoder = AutoConfig.from_pretrained(audio_encoder_pretrained_model_name_or_path, **kwargs_audio_encoder, return_unused_kwargs=True)
            if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                logger.info(f'Initializing {audio_encoder_pretrained_model_name_or_path} as an audio_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                encoder_config.is_decoder = False
                encoder_config.add_cross_attention = False
            kwargs_audio_encoder['config'] = encoder_config
        audio_encoder = AutoModel.from_pretrained(audio_encoder_pretrained_model_name_or_path, *model_args, **kwargs_audio_encoder)
    decoder = kwargs_decoder.pop('model', None)
    if decoder is None:
        if decoder_pretrained_model_name_or_path is None:
            raise ValueError('If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined.')
        if 'config' not in kwargs_decoder:
            decoder_config, kwargs_decoder = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True)
            if isinstance(decoder_config, MusicgenConfig):
                decoder_config = decoder_config.decoder
            if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                logger.info(f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers.")
                decoder_config.is_decoder = True
                decoder_config.add_cross_attention = True
            kwargs_decoder['config'] = decoder_config
        if kwargs_decoder['config'].is_decoder is False or kwargs_decoder['config'].add_cross_attention is False:
            logger.warning(f'Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_sub_models_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_sub_models_pretrained(...)`')
        decoder = MusicgenForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
    config = MusicgenConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config, **kwargs)
    return cls(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder, config=config)