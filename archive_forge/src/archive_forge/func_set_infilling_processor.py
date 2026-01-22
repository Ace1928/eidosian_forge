import os
from shutil import copyfile
from typing import List, Optional, Tuple
from tokenizers import normalizers, processors
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
from ...utils.versions import require_version
def set_infilling_processor(self, reset, suffix_first=False, add_special_tokens=True):
    """
        Updates the normalizer to make sure the prompt format for `infilling` is respected. The infilling format is the
        following: if suffix_first
            " <PRE> <SUF>{suf} <MID> {pre}"
        else:
            " <PRE> {pre} <SUF>{suf} <MID>"

        If `reset` is set to `True`, the `normalizer` and `post_processor` are reset to their "normal" behaviour, which
        is to add a prefix space for the normalizer, and add a `bos_token` to the input text for the `post_processor`.
        """
    if reset:
        self._tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend(prepend='▁'), normalizers.Replace(pattern=' ', content='▁')])
        self.update_post_processor()
        return
    self._tokenizer.normalizer = normalizers.Replace(pattern=' ', content='▁')
    pair = [self.bos_token] if self.add_bos_token and add_special_tokens else []
    special_tokens = [(self.bos_token, self.bos_token_id)] if self.add_bos_token and add_special_tokens else []
    if suffix_first:
        pair += [self.prefix_token, self.suffix_token, '$B', self.middle_token, '$A']
        special_tokens += [(self.prefix_token, self.prefix_id), (self.suffix_token, self.suffix_id), (self.middle_token, self.middle_id)]
    else:
        pair += [self.prefix_token, '$A', self.suffix_token, '$B', self.middle_token]
        special_tokens += [(self.prefix_token, self.prefix_id), (self.suffix_token, self.suffix_id), (self.middle_token, self.middle_id)]
    if self.add_eos_token and add_special_tokens:
        pair += [self.eos_token]
        special_tokens += [(self.eos_token, self.eos_token_id)]
    self._tokenizer.post_processor = processors.TemplateProcessing(single='$A', pair=pair, special_tokens=special_tokens)