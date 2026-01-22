import torch
import torch.nn as nn
from parlai.agents.transformer.modules import TransformerEncoder
def score_dialog(self, x_vecs, y_vecs):
    x_enc = self.x_dia_head(self.x_dia_encoder(x_vecs))
    if y_vecs.dtype == torch.float32:
        y_enc = y_vecs
    elif y_vecs.dtype == torch.int64:
        y_enc = self.encode_dia_y(y_vecs)
    else:
        raise Exception('Unsupported type for cands: {}'.format(type(y_vecs)))
    return self.score_similarity(x_enc, y_enc)