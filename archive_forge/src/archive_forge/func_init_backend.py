import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (
def init_backend(self, phonemizer_lang: str):
    """
        Initializes the backend.

        Args:
            phonemizer_lang (`str`): The language to be used.
        """
    requires_backends(self, 'phonemizer')
    from phonemizer.backend import BACKENDS
    self.backend = BACKENDS[self.phonemizer_backend](phonemizer_lang, language_switch='remove-flags')