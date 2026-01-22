import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def freeze_text_encoder(self):
    """
        Freeze the text (candidate) encoder.
        """
    self.text_encoder_frozen = True