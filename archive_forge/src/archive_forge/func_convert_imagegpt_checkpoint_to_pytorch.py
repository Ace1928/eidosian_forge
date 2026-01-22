import argparse
import torch
from transformers import ImageGPTConfig, ImageGPTForCausalLM, load_tf_weights_in_imagegpt
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
def convert_imagegpt_checkpoint_to_pytorch(imagegpt_checkpoint_path, model_size, pytorch_dump_folder_path):
    MODELS = {'small': (512, 8, 24), 'medium': (1024, 8, 36), 'large': (1536, 16, 48)}
    n_embd, n_head, n_layer = MODELS[model_size]
    config = ImageGPTConfig(n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model = ImageGPTForCausalLM(config)
    load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path)
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    print(f'Save PyTorch model to {pytorch_weights_dump_path}')
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f'Save configuration file to {pytorch_config_dump_path}')
    with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
        f.write(config.to_json_string())