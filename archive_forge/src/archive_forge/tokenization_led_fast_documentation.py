import json
from typing import Dict, List, Optional, Tuple, Union
from tokenizers import pre_tokenizers, processors
from ...tokenization_utils_base import AddedToken, BatchEncoding, EncodedInput
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, logging
from .tokenization_led import LEDTokenizer

        Create a mask from the two sequences passed to be used in a sequence-pair classification task. LED does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        