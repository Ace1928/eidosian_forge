from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.build_data import modelzoo_path
import torchtext.vocab as vocab
from parlai.utils.misc import TimeLogger
from collections import Counter, deque
import numpy as np
import os
import pickle
import torch
def get_word_sims(self, sent, sent_emb, dictionary):
    """
        Given a sentence and its Arora-style sentence embedding, compute the cosine
        similarities to it, for all words in the dictionary.

        Inputs:
          sent: string. Used only for caching lookup purposes.
          sent_emb: torch Tensor shape (glove_dim).
          dictionary: ParlAI dictionary

        Returns:
          sims: torch Tensor shape (vocab_size), containing the cosine sims.
        """
    if self.emb_matrix is None:
        self.get_emb_matrix(dictionary)
    if sent in self.cache_sent2sims:
        sims = self.cache_sent2sims[sent]
        return sims
    dotted = self.emb_matrix.dot(sent_emb)
    sent_emb_norm = np.linalg.norm(sent_emb)
    norms = np.multiply(self.emb_matrix_norm, sent_emb_norm)
    sims = np.divide(dotted, norms)
    sims = torch.tensor(sims)
    self.cache_sentqueue.append(sent)
    self.cache_sent2sims[sent] = sims
    if len(self.cache_sentqueue) > self.cache_limit:
        to_remove = self.cache_sentqueue.popleft()
        del self.cache_sent2sims[to_remove]
    assert len(self.cache_sent2sims) == len(self.cache_sentqueue)
    assert len(self.cache_sent2sims) <= self.cache_limit
    return sims