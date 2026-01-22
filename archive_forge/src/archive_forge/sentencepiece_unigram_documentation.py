import json
import os
from typing import Iterator, List, Optional, Union, Tuple
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.models import Unigram
from .base_tokenizer import BaseTokenizer

        Train the model using the given iterator

        Args:
            iterator (:obj:`Union[Iterator[str], Iterator[Iterator[str]]]`):
                Any iterator over strings or list of strings
            vocab_size (:obj:`int`):
                The size of the final vocabulary, including all tokens and alphabet.
            show_progress (:obj:`bool`):
                Whether to show progress bars while training.
            special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
                A list of special tokens the model should know of.
            initial_alphabet (:obj:`List[str]`, `optional`):
                A list of characters to include in the initial alphabet, even
                if not seen in the training dataset.
                If the strings contain more than one character, only the first one
                is kept.
            unk_token (:obj:`str`, `optional`):
                The unknown token to be used by the model.
            length (:obj:`int`, `optional`):
                The total number of sequences in the iterator. This is used to
                provide meaningful progress tracking
        