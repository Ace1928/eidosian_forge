import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
@torch.no_grad()
def generate_speech(self, input_values: torch.FloatTensor, speaker_embeddings: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, threshold: float=0.5, minlenratio: float=0.0, maxlenratio: float=20.0, vocoder: Optional[nn.Module]=None, output_cross_attentions: bool=False, return_output_lengths: bool=False) -> torch.FloatTensor:
    """
        Converts a raw speech waveform into a sequence of mel spectrograms, which are subsequently turned back into a
        speech waveform using a vocoder.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform.

                Values can be obtained by loading a *.flac* or *.wav* audio file into an array of type `List[float]` or
                a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install soundfile*). To prepare the array
                into `input_values`, the [`SpeechT5Processor`] should be used for padding and conversion into a tensor
                of type `torch.FloatTensor`. See [`SpeechT5Processor.__call__`] for details.
            speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
                Tensor containing the speaker embeddings.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            threshold (`float`, *optional*, defaults to 0.5):
                The generated sequence ends when the predicted stop token probability exceeds this value.
            minlenratio (`float`, *optional*, defaults to 0.0):
                Used to calculate the minimum required length for the output sequence.
            maxlenratio (`float`, *optional*, defaults to 20.0):
                Used to calculate the maximum allowed length for the output sequence.
            vocoder (`nn.Module`, *optional*, defaults to `None`):
                The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
                spectrogram.
            output_cross_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of the decoder's cross-attention layers.
            return_output_lengths (`bool`, *optional*, defaults to `False`):
                Whether or not to return the concrete spectrogram/waveform lengths.

        Returns:
            `tuple(torch.FloatTensor)` comprising various elements depending on the inputs:
            - when `return_output_lengths` is False
                - **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
                - **waveform** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(num_frames,)` -- The predicted speech waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
            - when `return_output_lengths` is True
                - **spectrograms** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrograms that
                are padded to the maximum length.
                - **spectrogram_lengths** (*optional*, returned when no `vocoder` is provided) `List[Int]` -- A list of
                all the concrete lengths for each spectrogram.
                - **waveforms** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, num_frames)` -- The predicted speech waveforms that are padded to the maximum length.
                - **waveform_lengths** (*optional*, returned when a `vocoder` is provided) `List[Int]` -- A list of all
                the concrete lengths for each waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
        """
    if speaker_embeddings is None:
        speaker_embeddings = torch.zeros((1, 512), device=input_values.device)
    return _generate_speech(self, input_values, speaker_embeddings, attention_mask, threshold, minlenratio, maxlenratio, vocoder, output_cross_attentions, return_output_lengths)