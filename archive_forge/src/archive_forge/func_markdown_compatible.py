import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def markdown_compatible(text: str) -> str:
    """
    Make text compatible with Markdown formatting.

    This function makes various text formatting adjustments to make it compatible with Markdown.

    Args:
        text (`str`):
            The input text to be made Markdown-compatible.

    Returns:
        `str`: The Markdown-compatible text.
    """
    text = re.sub('^\\(([\\d.]+[a-zA-Z]?)\\) \\\\\\[(.+?)\\\\\\]$', '\\[\\2 \\\\tag{\\1}\\]', text, flags=re.M)
    text = re.sub('^\\\\\\[(.+?)\\\\\\] \\(([\\d.]+[a-zA-Z]?)\\)$', '\\[\\1 \\\\tag{\\2}\\]', text, flags=re.M)
    text = re.sub('^\\\\\\[(.+?)\\\\\\] \\(([\\d.]+[a-zA-Z]?)\\) (\\\\\\[.+?\\\\\\])$', '\\[\\1 \\\\tag{\\2}\\] \\3', text, flags=re.M)
    text = text.replace('\\. ', '. ')
    text = text.replace('\\bm{', '\\mathbf{').replace('{\\\\bm ', '\\mathbf{')
    text = re.sub('\\\\mbox{ ?\\\\boldmath\\$(.*?)\\$}', '\\\\mathbf{\\1}', text)
    text = re.sub('((?:http|ftp|https):\\/\\/(?:[\\w_-]+(?:(?:\\.[\\w_-]+)+))(?:[\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-]))', '[\\1](\\1)', text)
    text = re.sub('```\\s*(.+?)\\s*```', '```\\n\\1\\n```', text, flags=re.S)
    return text