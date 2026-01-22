import json
import os
from typing import Optional, Tuple
import regex
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging, requires_backends

        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        