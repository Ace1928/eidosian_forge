from typing import TYPE_CHECKING
import torch
from ...utils import logging, recurse_getattr, recurse_setattr
def validate_bettertransformer(self):
    """
        A wrapper function to validate the `BetterTransformer` implementation. Implements most relevant checks
        that are present in: https://github.com/pytorch/pytorch/blob/0fc7de398636f4b53e6c3fde38b4e48a5ff5b37d/torch/nn/modules/transformer.py#L457-L475
        """
    if self.num_heads is None:
        raise ValueError('Number of heads not set for `BetterTransformer` integration.')
    if self.embed_dim is None:
        raise ValueError('Embedding dimension not set for `BetterTransformer` integration.')
    if self.norm2_eps is None or self.norm1_eps is None:
        raise ValueError('`norm2_eps` and `norm1_eps` not set for `BetterTransformer` integration.')
    if self.pos_emb_type is not None and self.pos_emb_type != 'absolute':
        raise ValueError(f'Positional embedding type {self.pos_emb_type} not supported for `BetterTransformer` integration')
    if self.norm1_eps != self.norm2_eps:
        raise ValueError('norm1_eps and norm2_eps must be equal for `BetterTransformer` integration.')
    if self.act_fn in USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS:
        logger.warning(f'Overridding {self.act_fn} activation with gelu. Use the transformed model at your own risk, the output logits could be significantly different.')
        self.act_fn = 'gelu'
    elif self.act_fn not in SUPPORTED_ACTIVATION_FUNCTIONS:
        raise ValueError(f'Activation function {self.act_fn} not supported for `BetterTransformer` integration.')
    self.use_gelu = self.act_fn == 'gelu' or self.act_fn == 'gelu_new'
    if self.num_heads % 2 == 1:
        raise ValueError(f'Number of heads {self.num_heads} is not supported for `BetterTransformer` integration. Number of heads must be even.')