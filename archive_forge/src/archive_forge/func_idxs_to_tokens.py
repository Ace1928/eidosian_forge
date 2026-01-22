from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def idxs_to_tokens(self, idxs: torch.LongTensor) -> List:
    """
        Map raw token IDs into corresponding tokens

        Args:
            idxs (LongTensor): raw token IDs generated from decoder

        Returns:
            List: tokens corresponding to the input IDs
        """
    return [self.tokens_dict.get_entry(idx.item()) for idx in idxs]