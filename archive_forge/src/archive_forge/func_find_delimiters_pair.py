import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def find_delimiters_pair(tokens, start_token, end_token):
    start_id = self.tokenizer.convert_tokens_to_ids(start_token)
    end_id = self.tokenizer.convert_tokens_to_ids(end_token)
    starting_positions = (tokens == start_id).nonzero(as_tuple=True)[0]
    ending_positions = (tokens == end_id).nonzero(as_tuple=True)[0]
    if torch.any(starting_positions) and torch.any(ending_positions):
        return (starting_positions[0], ending_positions[0])
    return (None, None)