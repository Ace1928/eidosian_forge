import torch
import torch.nn as nn
from parlai.agents.transformer.modules import TransformerEncoder
def score_similarity(self, context_h, cand_h):
    """
        Returns the dot product of encoded contexts and encoded candidates.
        """
    if self.opt['normalize_sent_emb']:
        context_h /= context_h.norm(2, dim=1, keepdim=True)
        cand_h /= cand_h.norm(2, dim=1, keepdim=True)
    if cand_h.dim() == 2:
        scores = torch.matmul(context_h, cand_h.t())
    elif cand_h.dim() == 3:
        scores = torch.bmm(context_h.unsqueeze(1), cand_h.transpose(1, 2)).squeeze(1)
    else:
        raise RuntimeError('Unexpected candidate dimensions {}'.format(cand_h.dim()))
    return scores