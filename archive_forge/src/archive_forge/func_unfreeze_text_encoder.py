import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def unfreeze_text_encoder(self):
    """
        Unfreeze the text (candidate) encoder.
        """
    self.text_encoder_frozen = False