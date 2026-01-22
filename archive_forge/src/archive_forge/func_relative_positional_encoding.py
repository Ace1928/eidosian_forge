from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
def relative_positional_encoding(self, qlen, klen, bsz=None):
    """create relative positional encoding."""
    freq_seq = tf.range(0, self.d_model, 2.0)
    inv_freq = 1 / 10000 ** (freq_seq / self.d_model)
    if self.attn_type == 'bi':
        beg, end = (klen, -qlen)
    elif self.attn_type == 'uni':
        beg, end = (klen, -1)
    else:
        raise ValueError(f'Unknown `attn_type` {self.attn_type}.')
    if self.bi_data:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        bwd_pos_seq = tf.range(-beg, -end, 1.0)
        if self.clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)
        if bsz is not None:
            if bsz % 2 != 0:
                raise ValueError(f'With bi_data, the batch size {bsz} should be divisible by 2')
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
        else:
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
        pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        if self.clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
    return pos_emb