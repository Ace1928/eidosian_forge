import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
def get_tcpgen_step_masks(self, yseqs, resettrie):
    seqlen = len(yseqs[0])
    batch_masks = yseqs.new_ones(len(yseqs), seqlen, len(self.char_list) + 1)
    p_gen_masks = []
    for i, yseq in enumerate(yseqs):
        new_tree = resettrie
        p_gen_mask = []
        for j, vy in enumerate(yseq):
            vy = vy.item()
            new_tree = new_tree[0]
            if vy in [self.blank_idx]:
                new_tree = resettrie
                p_gen_mask.append(0)
            elif self.char_list[vy].endswith('▁'):
                if vy in new_tree and new_tree[vy][0] != {}:
                    new_tree = new_tree[vy]
                else:
                    new_tree = resettrie
                p_gen_mask.append(0)
            elif vy not in new_tree:
                new_tree = [{}]
                p_gen_mask.append(1)
            else:
                new_tree = new_tree[vy]
                p_gen_mask.append(0)
            batch_masks[i, j, list(new_tree[0].keys())] = 0
        p_gen_masks.append(p_gen_mask + [1] * (seqlen - len(p_gen_mask)))
    p_gen_masks = torch.Tensor(p_gen_masks).to(yseqs.device).byte()
    return (batch_masks, p_gen_masks)