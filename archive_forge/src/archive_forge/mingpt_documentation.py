import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated

        input_embeds: [batch_size x seq_len x n_embed]
        attention_masks: [batch_size x seq_len], 0 don't attend, 1 attend
        