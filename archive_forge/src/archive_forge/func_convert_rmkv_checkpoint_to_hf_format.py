import argparse
import gc
import json
import os
import re
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, shard_checkpoint
def convert_rmkv_checkpoint_to_hf_format(repo_id, checkpoint_file, output_dir, size=None, tokenizer_file=None, push_to_hub=False, model_name=None):
    if tokenizer_file is None:
        print('No `--tokenizer_file` provided, we will use the default tokenizer.')
        vocab_size = 50277
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        vocab_size = len(tokenizer)
    tokenizer.save_pretrained(output_dir)
    possible_sizes = list(NUM_HIDDEN_LAYERS_MAPPING.keys())
    if size is None:
        for candidate in possible_sizes:
            if candidate in checkpoint_file:
                size = candidate
                break
        if size is None:
            raise ValueError('Could not infer the size, please provide it with the `--size` argument.')
    if size not in possible_sizes:
        raise ValueError(f'`size` should be one of {possible_sizes}, got {size}.')
    config = RwkvConfig(vocab_size=vocab_size, num_hidden_layers=NUM_HIDDEN_LAYERS_MAPPING[size], hidden_size=HIDEN_SIZE_MAPPING[size])
    config.save_pretrained(output_dir)
    model_file = hf_hub_download(repo_id, checkpoint_file)
    state_dict = torch.load(model_file, map_location='cpu')
    state_dict = convert_state_dict(state_dict)
    shards, index = shard_checkpoint(state_dict)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))
    if index is not None:
        save_index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)
        with open(save_index_file, 'w', encoding='utf-8') as f:
            content = json.dumps(index, indent=2, sort_keys=True) + '\n'
            f.write(content)
        print("Cleaning up shards. This may error with an OOM error, it this is the case don't worry you still have converted the model.")
        shard_files = list(shards.keys())
        del state_dict
        del shards
        gc.collect()
        for shard_file in shard_files:
            state_dict = torch.load(os.path.join(output_dir, shard_file))
            torch.save({k: v.cpu().clone() for k, v in state_dict.items()}, os.path.join(output_dir, shard_file))
    del state_dict
    gc.collect()
    if push_to_hub:
        if model_name is None:
            raise ValueError('Please provide a `model_name` to push the model to the Hub.')
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        model.push_to_hub(model_name, max_shard_size='2GB')
        tokenizer.push_to_hub(model_name)