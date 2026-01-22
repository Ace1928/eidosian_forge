from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
@property
def assistant(self):
    return f'{self.bos_token}assistant'