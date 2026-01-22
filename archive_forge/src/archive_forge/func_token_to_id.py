import os
from typing import List, Optional
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def token_to_id(self, token: str) -> int:
    return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))