import argparse
import json
import os
import torch
from transformers import LukeConfig, LukeModel, LukeTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken
def load_entity_vocab(entity_vocab_path):
    entity_vocab = {}
    with open(entity_vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            title, _ = line.rstrip().split('\t')
            entity_vocab[title] = index
    return entity_vocab