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
def get_glove_embs(self):
    """
        Loads torchtext GloVe embs from file and stores in self.tt_embs.
        """
    if not hasattr(self, 'glove_cache'):
        self.glove_cache = modelzoo_path(self.data_path, 'models:glove_vectors')
    print('Loading torchtext GloVe embs (for Arora sentence embs)...')
    self.tt_embs = vocab.GloVe(name=self.glove_name, dim=self.glove_dim, cache=self.glove_cache)
    print('Finished loading torchtext GloVe embs')