import dataclasses
import functools
import inspect
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available
from ...configuration_utils import _transformers_version
from ...utils import logging
def patched_speecht5_prenet_forward(self, input_values: torch.Tensor, speaker_embeddings: Optional[torch.Tensor]=None):
    inputs_embeds = input_values
    for layer in self.layers:
        inputs_embeds = torch.nn.functional.relu(layer(inputs_embeds))
        mask = torch.rand(inputs_embeds.shape, device=inputs_embeds.device) > self.config.speech_decoder_prenet_dropout
        inputs_embeds = inputs_embeds * mask / (1 - self.config.speech_decoder_prenet_dropout)
    inputs_embeds = self.final_layer(inputs_embeds)
    inputs_embeds = self.encode_positions(inputs_embeds)
    if speaker_embeddings is not None:
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings)
        speaker_embeddings = speaker_embeddings.unsqueeze(1)
        speaker_embeddings = speaker_embeddings.expand(-1, inputs_embeds.size(1), -1)
        inputs_embeds = torch.cat([inputs_embeds, speaker_embeddings], dim=-1)
        inputs_embeds = torch.nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))
    return inputs_embeds