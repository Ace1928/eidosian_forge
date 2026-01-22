import argparse
import os
import gluonnlp as nlp
import mxnet as mx
import numpy as np
import torch
from gluonnlp.base import get_home_dir
from gluonnlp.model.bert import BERTEncoder
from gluonnlp.model.utils import _load_vocab
from gluonnlp.vocab import Vocab
from packaging import version
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel, RobertaTokenizer
from transformers.models.bert.modeling_bert import (
from transformers.utils import logging
def check_and_map_params(hf_param, gluon_param):
    shape_hf = hf_param.shape
    gluon_param = to_torch(params[gluon_param])
    shape_gluon = gluon_param.shape
    assert shape_hf == shape_gluon, f'The gluon parameter {gluon_param} has shape {shape_gluon}, but expects shape {shape_hf} for Transformers'
    return gluon_param