import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
@property
def src_vocab_size(self):
    return len(self.encoder)