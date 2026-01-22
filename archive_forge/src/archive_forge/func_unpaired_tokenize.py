import os
from typing import List, Union
import tensorflow as tf
from tensorflow_text import BertTokenizer as BertTokenizerLayer
from tensorflow_text import FastBertTokenizer, ShrinkLongestTrimmer, case_fold_utf8, combine_segments, pad_model_inputs
from ...modeling_tf_utils import keras
from .tokenization_bert import BertTokenizer
def unpaired_tokenize(self, texts):
    if self.do_lower_case:
        texts = case_fold_utf8(texts)
    tokens = self.tf_tokenizer.tokenize(texts)
    return tokens.merge_dims(1, -1)