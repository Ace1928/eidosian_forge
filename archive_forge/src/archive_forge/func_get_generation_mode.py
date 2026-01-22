import copy
import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
def get_generation_mode(self, assistant_model: Optional['PreTrainedModel']=None) -> GenerationMode:
    """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
    if self.constraints is not None or self.force_words_ids is not None:
        generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
    elif self.num_beams == 1:
        if self.do_sample is False:
            if self.top_k is not None and self.top_k > 1 and (self.penalty_alpha is not None) and (self.penalty_alpha > 0):
                generation_mode = GenerationMode.CONTRASTIVE_SEARCH
            else:
                generation_mode = GenerationMode.GREEDY_SEARCH
        else:
            generation_mode = GenerationMode.SAMPLE
    elif self.num_beam_groups > 1:
        generation_mode = GenerationMode.GROUP_BEAM_SEARCH
    elif self.do_sample is True:
        generation_mode = GenerationMode.BEAM_SAMPLE
    else:
        generation_mode = GenerationMode.BEAM_SEARCH
    if assistant_model is not None or self.prompt_lookup_num_tokens is not None:
        if generation_mode in ('greedy_search', 'sample'):
            generation_mode = GenerationMode.ASSISTED_GENERATION
        else:
            raise ValueError("You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate is only supported with Greedy Search and Sample.")
    return generation_mode