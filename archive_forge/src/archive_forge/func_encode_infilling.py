import os
from logging import getLogger
from typing import List, Optional
from sentencepiece import SentencePieceProcessor
def encode_infilling(self, s: str) -> List[int]:
    """Encode a string without an implicit leading space."""
    return self.sp_model.encode('☺' + s)[2:]