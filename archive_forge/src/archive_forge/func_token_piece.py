import os
from logging import getLogger
from typing import List, Optional
from sentencepiece import SentencePieceProcessor
def token_piece(self, t: int) -> str:
    return self.sp_model.id_to_piece(t)