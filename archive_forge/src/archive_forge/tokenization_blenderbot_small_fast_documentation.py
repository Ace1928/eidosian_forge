from typing import List, Optional
from tokenizers import ByteLevelBPETokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_blenderbot_small import BlenderbotSmallTokenizer

        A very simple chat template that just adds whitespace between messages.
        