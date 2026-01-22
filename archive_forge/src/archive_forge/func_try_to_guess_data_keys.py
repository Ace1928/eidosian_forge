import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from .base import TaskProcessor
def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
    primary_key_name = None
    for name in column_names:
        if 'token' in name or 'text' in name or 'sentence' in name:
            primary_key_name = name
            break
    return {'primary': primary_key_name} if primary_key_name is not None else None