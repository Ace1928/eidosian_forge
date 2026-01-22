import os
from typing import List, Union
import tensorflow as tf
from tensorflow_text import BertTokenizer as BertTokenizerLayer
from tensorflow_text import FastBertTokenizer, ShrinkLongestTrimmer, case_fold_utf8, combine_segments, pad_model_inputs
from ...modeling_tf_utils import keras
from .tokenization_bert import BertTokenizer

        Instantiate a `TFBertTokenizer` from a pre-trained tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The name or path to the pre-trained tokenizer.

        Examples:

        ```python
        from transformers import TFBertTokenizer

        tf_tokenizer = TFBertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        ```
        