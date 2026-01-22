import argparse
import torch
from transformers import RemBertConfig, RemBertModel, load_tf_weights_in_rembert
from transformers.utils import logging
def convert_rembert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    config = RemBertConfig.from_json_file(bert_config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = RemBertModel(config)
    load_tf_weights_in_rembert(model, config, tf_checkpoint_path)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)