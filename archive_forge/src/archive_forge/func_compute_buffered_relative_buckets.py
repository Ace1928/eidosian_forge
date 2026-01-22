import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
def compute_buffered_relative_buckets(self, position_ids):
    batch_size, sequence_length = position_ids.shape
    position_ids = torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(1, 1)
    main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(self.num_buckets, self.relative_max_distance, position_ids)
    main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(batch_size, 1, 1)
    predict_relative_buckets = torch.cat([predict_relative_buckets[:, :sequence_length, :sequence_length], predict_relative_buckets[:, :sequence_length, self.max_target_positions:self.max_target_positions + sequence_length]], 2).repeat(batch_size, 1, 1)
    return (main_relative_buckets, predict_relative_buckets)