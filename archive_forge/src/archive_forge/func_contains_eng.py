import os
import fire
import re
from transformers import LlamaTokenizer
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
def contains_eng(text):
    eng_pattern = re.compile('[\\u0020-\\u007E]+')
    return True if eng_pattern.search(text) else False