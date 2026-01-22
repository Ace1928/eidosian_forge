from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
@property
def system(self):
    return f'{self.bos_token}system'