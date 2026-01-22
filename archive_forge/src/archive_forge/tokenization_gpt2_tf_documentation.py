import os
from typing import Dict, List, Union
import tensorflow as tf
from keras_nlp.tokenizers import BytePairTokenizer
from tensorflow_text import pad_model_inputs
from ...modeling_tf_utils import keras
from .tokenization_gpt2 import GPT2Tokenizer
Creates TFGPT2Tokenizer from configurations

        Args:
            config (Dict): Dictionary with keys such as stated in `get_config`.
        