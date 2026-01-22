from itertools import product
import math
import pytest
import torch
import transformers
from transformers import (
from tests.helpers import TRUE_FALSE, describe_dtype, id_formatter
def get_4bit_config():
    return BitsAndBytesConfig(load_in_4bit=True, load_in_8bit=False, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')