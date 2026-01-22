import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def post_process_generation(self, generation: Union[str, List[str]], fix_markdown: bool=True, num_workers: int=None) -> Union[str, List[str]]:
    """
        Postprocess a generated text or a list of generated texts.

        This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.

        Postprocessing is quite slow so it is recommended to use multiprocessing to speed up the process.

        Args:
            generation (Union[str, List[str]]):
                The generated text or a list of generated texts.
            fix_markdown (`bool`, *optional*, defaults to `True`):
                Whether to perform Markdown formatting fixes.
            num_workers (`int`, *optional*):
                Optional number of workers to pass to leverage multiprocessing (postprocessing several texts in
                parallel).

        Returns:
            Union[str, List[str]]: The postprocessed text or list of postprocessed texts.
        """
    requires_backends(self, ['nltk', 'levenshtein'])
    if isinstance(generation, list):
        if num_workers is not None and isinstance(num_workers, int):
            with Pool(num_workers) as p:
                return p.map(partial(self.post_process_single, fix_markdown=fix_markdown), generation)
        else:
            return [self.post_process_single(s, fix_markdown=fix_markdown) for s in generation]
    else:
        return self.post_process_single(generation, fix_markdown=fix_markdown)