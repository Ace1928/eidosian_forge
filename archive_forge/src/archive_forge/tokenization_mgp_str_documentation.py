import json
import os
from typing import Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
Converts an index (integer) in a token (str) using the vocab.