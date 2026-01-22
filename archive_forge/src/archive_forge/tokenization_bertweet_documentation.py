import html
import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple
import regex
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

        Args:
            text: str

        Returns: list(str) A tokenized list of strings; concatenating this list returns the original string if
        `preserve_case=False`
        