import os
from shutil import copyfile
from typing import Optional, Tuple
from tokenizers import processors
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
from ...utils.versions import require_version

        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        