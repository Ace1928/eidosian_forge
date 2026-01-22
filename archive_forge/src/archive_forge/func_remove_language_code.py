import json
import os
import re
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def remove_language_code(self, text: str):
    """Remove language codes like >>fr<< before sentencepiece"""
    match = self.language_code_re.match(text)
    code: list = [match.group(0)] if match else []
    return (code, self.language_code_re.sub('', text))