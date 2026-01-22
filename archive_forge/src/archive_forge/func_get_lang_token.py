import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece
from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer
from ...utils import logging
def get_lang_token(self, lang: str) -> str:
    return self.lang_code_to_token[lang]