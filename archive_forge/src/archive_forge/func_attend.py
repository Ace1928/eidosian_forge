from typing import Any, Dict, Optional, Tuple
import torch
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .biencoder import AddLabelFixedCandsTRA
from .modules import (
from .transformer import TransformerRankerAgent
def attend(self, attention_layer, queries, keys, values, mask):
    """
        Apply attention.

        :param attention_layer:
            nn.Module attention layer to use for the attention
        :param queries:
            the queries for attention
        :param keys:
            the keys for attention
        :param values:
            the values for attention
        :param mask:
            mask for the attention keys

        :return:
            the result of applying attention to the values, with weights computed
            wrt to the queries and keys.
        """
    if keys is None:
        keys = values
    if isinstance(attention_layer, PolyBasicAttention):
        return attention_layer(queries, keys, mask_ys=mask, values=values)
    elif isinstance(attention_layer, MultiHeadAttention):
        return attention_layer(queries, keys, values, mask)
    else:
        raise Exception('Unrecognized type of attention')