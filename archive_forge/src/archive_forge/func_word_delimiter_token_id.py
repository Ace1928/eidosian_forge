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
@word_delimiter_token_id.setter
def word_delimiter_token_id(self, value):
    self._word_delimiter_token = self.convert_tokens_to_ids(value)