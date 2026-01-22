import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def post_process_single(self, generation: str, fix_markdown: bool=True) -> str:
    """
        Postprocess a single generated text. Regular expressions used here are taken directly from the Nougat article
        authors. These expressions are commented for clarity and tested end-to-end in most cases.

        Args:
            generation (str): The generated text to be postprocessed.
            fix_markdown (bool, optional): Whether to perform Markdown formatting fixes. Default is True.

        Returns:
            str: The postprocessed text.
        """
    generation = re.sub('(?:\\n|^)#+ \\d*\\W? ?(.{100,})', '\\n\\1', generation)
    generation = generation.strip()
    generation = generation.replace('\n* [leftmargin=*]\n', '\n')
    generation = re.sub('^#+ (?:\\.?(?:\\d|[ixv])+)*\\s*(?:$|\\n\\s*)', '', generation, flags=re.M)
    lines = generation.split('\n')
    if lines[-1].startswith('#') and lines[-1].lstrip('#').startswith(' ') and (len(lines) > 1):
        logger.info('Likely hallucinated title at the end of the page: ' + lines[-1])
        generation = '\n'.join(lines[:-1])
    generation = truncate_repetitions(generation)
    generation = self.remove_hallucinated_references(generation)
    generation = re.sub('^\\* \\[\\d+\\](\\s?[A-W]\\.+\\s?){10,}.*$', '', generation, flags=re.M)
    generation = re.sub('^(\\* \\[\\d+\\])\\[\\](.*)$', '\\1\\2', generation, flags=re.M)
    generation = re.sub('(^\\w\\n\\n|\\n\\n\\w$)', '', generation)
    generation = re.sub('([\\s.,()])_([a-zA-Z0-9])__([a-zA-Z0-9]){1,3}_([\\s.,:()])', '\\1\\(\\2_{\\3}\\)\\4', generation)
    generation = re.sub('([\\s.,\\d])_([a-zA-Z0-9])_([\\s.,\\d;])', '\\1\\(\\2\\)\\3', generation)
    generation = re.sub('(\\nFootnote .*?:) (?:footnotetext|thanks):\\W*(.*(?:\\n\\n|$))', '\\1 \\2', generation)
    generation = re.sub('\\[FOOTNOTE:.+?\\](.*?)\\[ENDFOOTNOTE\\]', '', generation)
    generation = normalize_list_like_lines(generation)
    if generation.endswith(('.', '}')):
        generation += '\n\n'
    if re.match('[A-Z0-9,;:]$', generation):
        generation += ' '
    elif generation.startswith(('#', '**', '\\begin')):
        generation = '\n\n' + generation
    elif generation.split('\n')[-1].startswith(('#', 'Figure', 'Table')):
        generation = generation + '\n\n'
    else:
        try:
            last_word = generation.split(' ')[-1]
            if last_word in nltk.corpus.words.words():
                generation += ' '
        except LookupError:
            generation += ' '
    generation = self.correct_tables(generation)
    generation = generation.replace('\\begin{array}[]{', '\\begin{array}{')
    generation = re.sub('\\\\begin{tabular}{([clr ]){2,}}\\s*[& ]*\\s*(\\\\\\\\)? \\\\end{tabular}', '', generation)
    generation = re.sub('(\\*\\*S\\. A\\. B\\.\\*\\*\\n+){2,}', '', generation)
    generation = re.sub('^#+( [\\[\\d\\w])?$', '', generation, flags=re.M)
    generation = re.sub('^\\.\\s*$', '', generation, flags=re.M)
    generation = re.sub('\\n{3,}', '\n\n', generation)
    if fix_markdown:
        return markdown_compatible(generation)
    else:
        return generation