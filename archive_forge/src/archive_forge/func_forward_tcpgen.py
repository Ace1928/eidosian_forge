import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
def forward_tcpgen(self, targets, ptrdist_mask, source_encodings):
    tcpgen_dist = None
    if self.DBaverage and self.deepbiasing:
        hptr = self.biasingemb(1 - ptrdist_mask[:, :, :-1].float()).unsqueeze(1)
    else:
        query_char = self.predictor.embedding(targets)
        query_char = self.Qproj_char(query_char).unsqueeze(1)
        query_acoustic = self.Qproj_acoustic(source_encodings).unsqueeze(2)
        query = query_char + query_acoustic
        hptr, tcpgen_dist = self.get_tcpgen_distribution(query, ptrdist_mask)
    return (hptr, tcpgen_dist)