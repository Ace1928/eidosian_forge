from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
def tf_to_pt_weight_rename(self, tf_weight):
    if self.config.tie_word_embeddings and 'crit.out_layers' in tf_weight:
        return (tf_weight, tf_weight.replace('crit.out_layers', 'transformer.word_emb.emb_layers'))
    elif self.config.tie_projs and 'crit.out_projs' in tf_weight:
        for i, tie_proj in enumerate(self.config.tie_projs):
            if tie_proj and self.config.div_val == 1 and (self.config.d_model != self.config.d_embed):
                return (tf_weight, tf_weight.replace(f'crit.out_projs.{i}', 'transformer.word_emb.emb_projs.0'))
            elif tie_proj and self.config.div_val != 1:
                return (tf_weight, tf_weight.replace('crit.out_projs', 'transformer.word_emb.emb_projs'))
    else:
        return (tf_weight,)