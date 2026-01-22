import os
from logging import getLogger
from typing import List, Optional
from sentencepiece import SentencePieceProcessor
def decode_infilling(self, t: List[int]) -> str:
    """Decode a string without an implicit leading space."""
    return self.sp_model.decode([self.sp_model.piece_to_id('☺')] + t)[1:]