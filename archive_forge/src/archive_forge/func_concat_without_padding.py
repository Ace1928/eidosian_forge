import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def concat_without_padding(text_idx, cand_idx, use_cuda, null_idx=0):
    """
    Concatenate two right padded tensors and move padding to the right.

    For example,
        if text_idx = [[1, 2, 3, 4, 0, 0  ]]
        and cand_idx = [[5, 6, 7, 8, 0, 0 ]]:
    Then result = (tokens, segments) where
        tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]]
        segments = [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
    """
    assert text_idx.size(0) == cand_idx.size(0)
    assert len(text_idx.size()) == 2
    assert len(cand_idx.size()) == 2
    segments_idx = [0, 1]
    text_idx = text_idx.cpu()
    cand_idx = cand_idx.cpu()
    cand_len = cand_idx.size(1)
    concat_len = text_idx.size(1) + cand_idx.size(1)
    tokens = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    segments = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    for i in range(len(tokens)):
        non_nuls = torch.sum(text_idx[i, :] != null_idx)
        tokens[i, 0:non_nuls] = text_idx[i, 0:non_nuls]
        segments[i, 0:non_nuls] = segments_idx[0]
        tokens[i, non_nuls:non_nuls + cand_len] = cand_idx[i, :]
        segments[i, non_nuls:non_nuls + cand_len] = segments_idx[1]
    if use_cuda:
        tokens = tokens.cuda()
        segments = segments.cuda()
    return (tokens, segments)