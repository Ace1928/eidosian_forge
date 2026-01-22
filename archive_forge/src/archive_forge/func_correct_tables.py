import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def correct_tables(self, generation: str) -> str:
    """
        Takes a generated string and fixes tables/tabulars to make them match the markdown format needed.

        Args:
            generation (str): The generated text to be postprocessed.

        Returns:
            str: The postprocessed text.

        Example:

        ```python
        correct_tables("\\begin{table} \\begin{tabular}{l l} & \\ \\end{tabular} \\end{table}")
        "\\begin{table}
\\begin{tabular}{l l} & \\ \\end{tabular}
\\end{table}"
        ```
        """
    for l in generation.split('\n'):
        if l.count('\\begin{tabular}') > 15 or l.count('\\multicolumn') > 60 or l.count('&') > 400:
            generation = generation.replace(l, '')
    generation = generation.replace('\\begin{table} \\begin{tabular}', '\\begin{table}\n\\begin{tabular}')
    generation = generation.replace('\\end{tabular} \\end{table}', '\\end{tabular}\n\\end{table}')
    generation = generation.replace('\\end{table} Tab', '\\end{table}\nTab')
    generation = re.sub('(^.+)\\\\begin{tab', '\\1\\n\\\\begin{tab', generation, flags=re.M)
    generation = generation.replace('\\begin{tabular}{l l}  & \\\\ \\end{tabular}', '')
    generation = generation.replace('\\begin{tabular}{}\n\n\\end{tabular}', '')
    return generation