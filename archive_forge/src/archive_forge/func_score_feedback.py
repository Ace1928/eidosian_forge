import torch
import torch.nn as nn
from parlai.agents.transformer.modules import TransformerEncoder
def score_feedback(self, x_vecs, y_vecs):
    x_enc = self.x_fee_head(self.x_fee_encoder(x_vecs))
    y_enc = self.y_fee_head(self.y_fee_encoder(y_vecs))
    return self.score_similarity(x_enc, y_enc)