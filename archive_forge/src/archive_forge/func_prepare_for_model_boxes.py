import os
import re
import warnings
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging
@add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
def prepare_for_model_boxes(self, text: Union[TextInput, PreTokenizedInput], text_pair: Optional[PreTokenizedInput]=None, boxes: Optional[List[List[int]]]=None, word_labels: Optional[List[int]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, prepend_batch_axis: bool=False, **kwargs) -> BatchEncoding:
    """
        Prepares a sequence or a pair of sequences so that it can be used by the model. It adds special tokens,
        truncates sequences if overflowing while taking into account the special tokens and manages a moving window
        (with user defined stride) for overflowing tokens.

        Word-level `boxes` are turned into token-level `bbox`. If provided, word-level `word_labels` are turned into
        token-level `labels`. The word label is used for the first token of the word, while remaining tokens are
        labeled with -100, such that they will be ignored by the loss function.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """
    padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, **kwargs)
    tokens = []
    pair_tokens = []
    token_boxes = []
    pair_token_boxes = []
    labels = []
    if text_pair is None:
        if word_labels is None:
            for word, box in zip(text, boxes):
                if len(word) < 1:
                    continue
                word_tokens = self.tokenize(word)
                tokens.extend(word_tokens)
                token_boxes.extend([box] * len(word_tokens))
        else:
            for word, box, label in zip(text, boxes, word_labels):
                if len(word) < 1:
                    continue
                word_tokens = self.tokenize(word)
                tokens.extend(word_tokens)
                token_boxes.extend([box] * len(word_tokens))
                if self.only_label_first_subword:
                    labels.extend([label] + [self.pad_token_label] * (len(word_tokens) - 1))
                else:
                    labels.extend([label] * len(word_tokens))
    else:
        tokens = self.tokenize(text)
        token_boxes = [self.pad_token_box for _ in range(len(tokens))]
        for word, box in zip(text_pair, boxes):
            if len(word) < 1:
                continue
            word_tokens = self.tokenize(word)
            pair_tokens.extend(word_tokens)
            pair_token_boxes.extend([box] * len(word_tokens))
    ids = self.convert_tokens_to_ids(tokens)
    pair_ids = self.convert_tokens_to_ids(pair_tokens) if pair_tokens else None
    pair = bool(pair_ids is not None)
    len_ids = len(ids)
    len_pair_ids = len(pair_ids) if pair else 0
    total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
    overflowing_tokens = []
    overflowing_token_boxes = []
    overflowing_labels = []
    if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and (total_len > max_length):
        ids, token_boxes, pair_ids, pair_token_boxes, labels, overflowing_tokens, overflowing_token_boxes, overflowing_labels = self.truncate_sequences(ids, token_boxes, pair_ids=pair_ids, pair_token_boxes=pair_token_boxes, labels=labels, num_tokens_to_remove=total_len - max_length, truncation_strategy=truncation_strategy, stride=stride)
    if return_token_type_ids and (not add_special_tokens):
        raise ValueError('Asking to return token_type_ids while setting add_special_tokens to False results in an undefined behavior. Please set add_special_tokens to True or set return_token_type_ids to None.')
    if return_token_type_ids is None:
        return_token_type_ids = 'token_type_ids' in self.model_input_names
    if return_attention_mask is None:
        return_attention_mask = 'attention_mask' in self.model_input_names
    encoded_inputs = {}
    if return_overflowing_tokens:
        encoded_inputs['overflowing_tokens'] = overflowing_tokens
        encoded_inputs['overflowing_token_boxes'] = overflowing_token_boxes
        encoded_inputs['overflowing_labels'] = overflowing_labels
        encoded_inputs['num_truncated_tokens'] = total_len - max_length
    if add_special_tokens:
        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        token_boxes = token_boxes + [self.sep_token_box]
        if pair_token_boxes:
            pair_token_boxes = pair_token_boxes + [self.sep_token_box]
        if labels:
            labels = labels + [self.pad_token_label]
    else:
        sequence = ids + pair_ids if pair else ids
        token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
    encoded_inputs['input_ids'] = sequence
    encoded_inputs['bbox'] = token_boxes + pair_token_boxes
    if return_token_type_ids:
        encoded_inputs['token_type_ids'] = token_type_ids
    if return_special_tokens_mask:
        if add_special_tokens:
            encoded_inputs['special_tokens_mask'] = self.get_special_tokens_mask(ids, pair_ids)
        else:
            encoded_inputs['special_tokens_mask'] = [0] * len(sequence)
    if labels:
        encoded_inputs['labels'] = labels
    self._eventual_warn_about_too_long_sequence(encoded_inputs['input_ids'], max_length, verbose)
    if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
        encoded_inputs = self.pad(encoded_inputs, max_length=max_length, padding=padding_strategy.value, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
    if return_length:
        encoded_inputs['length'] = len(encoded_inputs['input_ids'])
    batch_outputs = BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)
    return batch_outputs