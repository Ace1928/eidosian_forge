from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.torch import padded_3d
from parlai.zoo.bert.build import download
from .bert_dictionary import BertDictionaryAgent
from .helpers import (
import os
import torch
from tqdm import tqdm
def vectorize_fixed_candidates(self, cands_batch):
    """
        Override from TorchRankerAgent.
        """
    return [self._vectorize_text(cand, add_start=True, add_end=True, truncate=self.label_truncate) for cand in cands_batch]