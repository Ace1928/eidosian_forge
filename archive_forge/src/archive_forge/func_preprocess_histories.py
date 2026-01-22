import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
def preprocess_histories(self, max_coarse_history: int, semantic_to_coarse_ratio: int, batch_size: int, semantic_generation_config: int, codebook_size: int, history_prompt: Optional[Dict[str, torch.Tensor]]=None):
    """
        Preprocess the optional `Bark` speaker prompts before `self.generate`.

        Args:
            max_coarse_history (`int`):
                Maximum size of coarse tokens used.
            semantic_to_coarse_ratio (`int`):
                Ratio of semantic to coarse frequency
            batch_size (`int`):
                Batch size, i.e the number of samples.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            codebook_size (`int`):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`):
                Optional `Bark` speaker prompt.
        Returns: Returns:
            `tuple(torch.FloatTensor)`:
            - **x_semantic_history** (`torch.FloatTensor` -- Processed semantic speaker prompt.
            - **x_coarse_history** (`torch.FloatTensor`) -- Processed coarse speaker prompt.
        """
    if history_prompt is not None:
        x_semantic_history = torch.repeat_interleave(history_prompt['semantic_prompt'][None], batch_size, dim=0)
        x_coarse_history = history_prompt['coarse_prompt'].clone()
        if codebook_size is not None:
            for n in range(1, x_coarse_history.shape[0]):
                x_coarse_history[n, :] += codebook_size * n
        x_coarse_history = torch.transpose(x_coarse_history, 0, 1).view(-1)
        x_coarse_history = x_coarse_history + semantic_generation_config.semantic_vocab_size
        x_coarse_history = torch.repeat_interleave(x_coarse_history[None], batch_size, dim=0)
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
        n_semantic_hist_provided = min([max_semantic_history, x_semantic_history.shape[1] - x_semantic_history.shape[1] % 2, int(np.floor(x_coarse_history.shape[1] / semantic_to_coarse_ratio))])
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[:, -n_semantic_hist_provided:].int()
        x_coarse_history = x_coarse_history[:, -n_coarse_hist_provided:].int()
        x_coarse_history = x_coarse_history[:, :-2]
    else:
        x_semantic_history = torch.tensor([[]] * batch_size, dtype=torch.int).to(self.device)
        x_coarse_history = torch.tensor([[]] * batch_size, dtype=torch.int).to(self.device)
    return (x_semantic_history, x_coarse_history)