from collections import OrderedDict
from torch import nn
from vllm.utils import LRUCache
from vllm.lora.utils import (parse_fine_tuned_lora_name, replace_submodule)
def test_parse_fine_tuned_lora_name():
    fixture = {('base_model.model.lm_head.lora_A.weight', 'lm_head', True), ('base_model.model.lm_head.lora_B.weight', 'lm_head', False), ('base_model.model.model.embed_tokens.lora_embedding_A', 'model.embed_tokens', True), ('base_model.model.model.embed_tokens.lora_embedding_B', 'model.embed_tokens', False), ('base_model.model.model.layers.9.mlp.down_proj.lora_A.weight', 'model.layers.9.mlp.down_proj', True), ('base_model.model.model.layers.9.mlp.down_proj.lora_B.weight', 'model.layers.9.mlp.down_proj', False)}
    for name, module_name, is_lora_a in fixture:
        assert (module_name, is_lora_a) == parse_fine_tuned_lora_name(name)