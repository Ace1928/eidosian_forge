import argparse
import pickle
import numpy as np
import torch
from torch import nn
from transformers import ReformerConfig, ReformerModelWithLMHead
from transformers.utils import logging
def convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file, pytorch_dump_path):
    config = ReformerConfig.from_json_file(config_file)
    print(f'Building PyTorch model from configuration: {config}')
    model = ReformerModelWithLMHead(config)
    with open(trax_model_pkl_path, 'rb') as f:
        model_weights = pickle.load(f)['weights']
    set_model_weights_in_torch(model_weights, model, config.hidden_size)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    torch.save(model.state_dict(), pytorch_dump_path)