import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
def get_tcpgen_distribution(self, query, ptrdist_mask):
    keyvalues = torch.cat([self.predictor.embedding.weight.data, self.ooKBemb.weight], dim=0)
    keyvalues = self.dropout_tcpgen(self.Kproj(keyvalues))
    tcpgendist = torch.einsum('ntuj,ij->ntui', query, keyvalues)
    tcpgendist = tcpgendist / math.sqrt(query.size(-1))
    ptrdist_mask = ptrdist_mask.unsqueeze(1).repeat(1, tcpgendist.size(1), 1, 1)
    tcpgendist.masked_fill_(ptrdist_mask.bool(), -1000000000.0)
    tcpgendist = torch.nn.functional.softmax(tcpgendist, dim=-1)
    hptr = torch.einsum('ntui,ij->ntuj', tcpgendist[:, :, :, :-1], keyvalues[:-1, :])
    return (hptr, tcpgendist)