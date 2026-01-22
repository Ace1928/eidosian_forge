import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def update_codebook(self, hidden_states, latent_states):
    mu, codebook_width, nb_discrete_codes = (self.mu, self.codebook_width, self.nb_discrete_codes)
    with torch.no_grad():
        latent_states_onehot = torch.zeros(nb_discrete_codes, hidden_states.shape[0], device=hidden_states.device)
        latent_states_onehot.scatter_(0, latent_states.view(1, hidden_states.shape[0]), 1)
        _codebook_sum = torch.matmul(latent_states_onehot, hidden_states)
        _codebook_elem = latent_states_onehot.sum(dim=-1)
        codes = self._tile(hidden_states)
        _random_codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]
        old_codebook = self.codebook
        self.codebook_sum = mu * self.codebook_sum + (1.0 - mu) * _codebook_sum
        self.codebook_elem = mu * self.codebook_elem + (1.0 - mu) * _codebook_elem
        usage = (self.codebook_elem.view(nb_discrete_codes, 1) >= self.threshold).float()
        norm_code = self.codebook_sum.view(nb_discrete_codes, codebook_width) / self.codebook_elem.view(nb_discrete_codes, 1)
        self.codebook = usage * norm_code + (1 - usage) * _random_codebook
        _codebook_prob = _codebook_elem / torch.sum(_codebook_elem)
        entropy = -torch.sum(_codebook_prob * torch.log(_codebook_prob + 1e-08))
        used_curr = (_codebook_elem >= self.threshold).sum()
        usage = torch.sum(usage)
        dk = torch.norm(self.codebook - old_codebook) / np.sqrt(np.prod(old_codebook.shape))
    return {'entropy': entropy, 'used_curr': used_curr, 'usage': usage, 'dk': dk}