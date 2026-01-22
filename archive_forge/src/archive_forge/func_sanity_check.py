import argparse
import json
import os
import tensorstore as ts
import torch
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from tensorflow.io import gfile
from transformers.modeling_utils import dtype_byte_size
from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import convert_file_size_to_int
def sanity_check():
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration, T5Tokenizer
    config = SwitchTransformersConfig.from_pretrained('google/switch-base-8')
    config.save_pretrained('/home/arthur_huggingface_co/transformers/switch_converted')
    model = SwitchTransformersForConditionalGeneration.from_pretrained('/home/arthur_huggingface_co/transformers/switch_converted', device_map='auto')
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    text = 'A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>.'
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    out = model.generate(input_ids, decoder_start_token_id=0)
    print(tokenizer.decode(out[0]))