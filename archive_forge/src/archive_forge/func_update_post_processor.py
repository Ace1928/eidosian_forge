import os
from shutil import copyfile
from typing import Optional, Tuple
from tokenizers import processors
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
from ...utils.versions import require_version
def update_post_processor(self):
    """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
    bos = self.bos_token
    bos_token_id = self.bos_token_id
    if bos is None and self.add_bos_token:
        raise ValueError('add_bos_token = True but bos_token = None')
    eos = self.eos_token
    eos_token_id = self.eos_token_id
    if eos is None and self.add_eos_token:
        raise ValueError('add_eos_token = True but eos_token = None')
    single = f'{(bos + ':0 ' if self.add_bos_token else '')}$A:0{(' ' + eos + ':0' if self.add_eos_token else '')}'
    pair = f'{single}{(' ' + bos + ':1' if self.add_bos_token else '')} $B:1{(' ' + eos + ':1' if self.add_eos_token else '')}'
    special_tokens = []
    if self.add_bos_token:
        special_tokens.append((bos, bos_token_id))
    if self.add_eos_token:
        special_tokens.append((eos, eos_token_id))
    self._tokenizer.post_processor = processors.TemplateProcessing(single=single, pair=pair, special_tokens=special_tokens)