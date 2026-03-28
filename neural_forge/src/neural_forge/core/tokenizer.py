from __future__ import annotations

from typing import List, Union
import torch

class ByteLevelTokenizer:
    """
    A pure Eidosian byte-level tokenizer. No assumptions, no language-specific rules.
    """
    def __init__(self):
        self.vocab_size = 256 # Standard ASCII/Byte range

    def encode(self, text: str) -> torch.Tensor:
        """Convert text into a tensor of byte values."""
        bytes_data = text.encode("utf-8")
        return torch.tensor([list(bytes_data)], dtype=torch.long)

    def decode(self, tokens: Union[torch.Tensor, List[int]]) -> str:
        """Convert tokens back into a UTF-8 string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()[0]
        return bytes(tokens).decode("utf-8", errors="replace")
