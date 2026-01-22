import os, copy, types, gc, sys
import numpy as np
from prompt_toolkit import prompt
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']