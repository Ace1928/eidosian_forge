import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import BatchEncoding
from ...utils import PaddingStrategy, logging
def batch_encode_candidates(self, text, **kwargs):
    """
        Encode a batch of text or text pair. This method is similar to regular __call__ method but has the following
        differences:

            1. Handle additional num_candidate axis. (batch_size, num_candidates, text)
            2. Always pad the sequences to *max_length*.
            3. Must specify *max_length* in order to stack packs of candidates into a batch.

            - single sequence: `[CLS] X [SEP]`
            - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            text (`List[List[str]]`):
                The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,
                num_candidates, text).
            text_pair (`List[List[str]]`, *optional*):
                The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,
                num_candidates, text).
            **kwargs:
                Keyword arguments of the __call__ method.

        Returns:
            [`BatchEncoding`]: Encoded text or text pair.

        Example:

        ```python
        >>> from transformers import RealmTokenizer

        >>> # batch_size = 2, num_candidates = 2
        >>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

        >>> tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-encoder")
        >>> tokenized_text = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
        ```"""
    kwargs['padding'] = PaddingStrategy.MAX_LENGTH
    batch_text = text
    batch_text_pair = kwargs.pop('text_pair', None)
    return_tensors = kwargs.pop('return_tensors', None)
    output_data = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    for idx, candidate_text in enumerate(batch_text):
        if batch_text_pair is not None:
            candidate_text_pair = batch_text_pair[idx]
        else:
            candidate_text_pair = None
        encoded_candidates = super().__call__(candidate_text, candidate_text_pair, return_tensors=None, **kwargs)
        encoded_input_ids = encoded_candidates.get('input_ids')
        encoded_attention_mask = encoded_candidates.get('attention_mask')
        encoded_token_type_ids = encoded_candidates.get('token_type_ids')
        if encoded_input_ids is not None:
            output_data['input_ids'].append(encoded_input_ids)
        if encoded_attention_mask is not None:
            output_data['attention_mask'].append(encoded_attention_mask)
        if encoded_token_type_ids is not None:
            output_data['token_type_ids'].append(encoded_token_type_ids)
    output_data = {key: item for key, item in output_data.items() if len(item) != 0}
    return BatchEncoding(output_data, tensor_type=return_tensors)