import argparse
import json
import os
import re
import sys
import types
import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script

    """
    os.makedirs(args.save_path, exist_ok=True)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    try:
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    except ModuleNotFoundError:
        print('Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.')
        exit(1)
    sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith('pytorch_model')]
    if len(sub_dirs) == 1:
        checkpoint_name = 'pytorch_model.bin'
        state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location='cpu')
    else:
        num_checkpoints = len(sub_dirs) - 1
        state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)
    config = GPT2Config.from_pretrained(args.load_path)
    tracker_filepath = os.path.join(args.save_path, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, 'w') as f:
        f.write('release')
    release_dir = os.path.join(args.save_path, 'release')
    os.makedirs(release_dir, exist_ok=True)
    megatron_args = {'orig_vocab_size': config.vocab_size, 'max_position_embeddings': config.n_positions, 'hidden_size': config.n_embd, 'num_layers': config.n_layer, 'num_attention_heads': config.n_head, 'ffn_hidden_size': config.n_inner, 'tensor_model_parallel_size': args.target_tensor_model_parallel_size, 'pipeline_model_parallel_size': args.target_pipeline_model_parallel_size, 'data_parallel_size': args.target_data_parallel_size, 'make_vocab_size_divisible_by': args.make_vocab_size_divisible_by, 'rank': 0, 'tokenizer_type': 'GPT2BPETokenizer'}
    if config.activation_function == 'gelu':
        megatron_args['bias_gelu_fusion'] = False
        megatron_args['openai_gelu'] = False
    elif config.activation_function == 'gelu_fast':
        megatron_args['bias_gelu_fusion'] = True
        megatron_args['openai_gelu'] = False
    elif config.activation_function == 'gelu_new':
        megatron_args['bias_gelu_fusion'] = False
        megatron_args['openai_gelu'] = True
    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)
    if args.target_params_dtype == 'fp16':
        dtype = torch.float16
    elif args.target_params_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, 'params_dtype', dtype)
    dummy_optim_state_dict = {}
    dummy_optim_state_dict['optimizer'] = {'step': 0, 'param_groups': [{'lr': 0.0, 'beta1': 0.0, 'beta2': 0.0, 'eps': 0.0, 'weight_decay': 0.0, 'correct_bias': False, 'params': []}]}
    if args.use_distributed_optimizer:
        for i in range(args.target_pipeline_model_parallel_size):
            for j in range(args.target_tensor_model_parallel_size):
                for k in range(args.target_data_parallel_size):
                    if args.target_pipeline_model_parallel_size == 1:
                        checkpoint_dir = f'mp_rank_{j:02d}_{k:03d}'
                    else:
                        checkpoint_dir = f'mp_rank_{j:02d}_{i:03d}_{k:03d}'
                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(dummy_optim_state_dict, os.path.join(checkpoint_dir, 'optim.pt'))
    print('Converting')
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})
    print('converting embedding layer')
    pos_embedding = state_dict['transformer.wpe.weight'].to(dtype)
    word_embedding = state_dict['transformer.wte.weight'].to(dtype)
    orig_vocab_size = config.vocab_size
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
    setattr(margs, 'padded_vocab_size', padded_vocab_size)
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = word_embedding[0:padded_vocab_size, :]
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size
        full_word_embed = torch.cat((word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
    else:
        full_word_embed = word_embedding
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        pos_emb_dict = get_element_from_dict_by_path(output_state_dict[i], 'model.language_model.embedding.position_embeddings')
        pos_emb_dict['weight'] = pos_embedding
        word_emb_dict = get_element_from_dict_by_path(output_state_dict[i], 'model.language_model.embedding.word_embeddings')
        word_emb_dict['weight'] = out_word_embed[i].clone()
    print('converting transformer layers')
    if config.num_attention_heads % args.target_tensor_model_parallel_size != 0:
        raise ValueError(f'Number of attention heads ({config.num_attention_heads}) must be divisible by number of tensor parallelism ({args.target_tensor_model_parallel_size})')
    if config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(f'Number of layers ({config.num_hidden_layers}) must be divisible by number of pipeline parallelism ({args.target_pipeline_model_parallel_size})')
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size
    layer_re = re.compile('transformer.h\\.(\\d+)\\.([a-z0-9_.]+)\\.([a-z]+)')
    heads = config.n_head
    hidden_size_per_head = config.n_embd // config.n_head
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})
        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [layer_name for layer_name in state_dict.keys() if layer_name.startswith(f'transformer.h.{pp_layer_id}.')]
            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                if m is None:
                    break
                _ = int(m.group(1))
                op_name = m.group(2)
                weight_or_bias = m.group(3)
                params = state_dict[layer_name].to(dtype)
                if op_name.startswith('ln'):
                    out_name = 'input_layernorm' if op_name.endswith('1') else 'post_attention_layernorm'
                    layer_name = f'layers.{layer}.{out_name}.{weight_or_bias}'
                elif op_name.startswith('attn.c_attn') and weight_or_bias == 'weight':
                    params = params.transpose(0, 1).contiguous()
                    params = transformers_to_megatron_fix_query_key_value_ordering(params, 3.0, 3, heads, hidden_size_per_head)
                    layer_name = f'layers.{layer}.self_attention.query_key_value.{weight_or_bias}'
                elif op_name.startswith('attn.c_attn') and weight_or_bias == 'bias':
                    params = transformers_to_megatron_fix_query_key_value_ordering(params, 3.0, 3, heads, hidden_size_per_head)
                    layer_name = f'layers.{layer}.self_attention.query_key_value.{weight_or_bias}'
                elif weight_or_bias == 'weight':
                    out_name = transformers_to_megatron.get(op_name, None)
                    if out_name is None:
                        continue
                    params = params.transpose(0, 1)
                    layer_name = f'layers.{layer}.{out_name}.{weight_or_bias}'
                elif weight_or_bias == 'bias':
                    out_name = transformers_to_megatron.get(op_name, None)
                    if out_name is None:
                        continue
                    layer_name = f'layers.{layer}.{out_name}.{weight_or_bias}'
                else:
                    continue
                if op_name + '.' + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name in ['attn.c_proj', 'mlp.c_proj'] else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], 'model.language_model.encoder')
                    params_dict[layer_name] = params[i].clone() if op_name + '.' + weight_or_bias in tensor_parallel_params else params
        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            for weight_or_bias in ['weight', 'bias']:
                params = state_dict[f'transformer.ln_f.{weight_or_bias}'].to(dtype)
                layer_name = f'final_layernorm.{weight_or_bias}'
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], 'model.language_model.encoder')
                    params_dict[layer_name] = params
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], 'model.word_embeddings_for_head')
                params_dict['weight'] = out_word_embed[i].clone()
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]['checkpoint_version'] = 3.0
            output_state_dict[tp_rank]['args'] = margs
            checkpoint_dir = f'mp_rank_{tp_rank:02d}' if args.target_pipeline_model_parallel_size == 1 else f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'
            if args.use_distributed_optimizer:
                checkpoint_name = 'model_rng.pt'
            else:
                checkpoint_name = 'model_optim_rng.pt'
                output_state_dict[tp_rank]['optimizer'] = dummy_optim_state_dict['optimizer']
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(f'Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank {pp_rank}:')
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)