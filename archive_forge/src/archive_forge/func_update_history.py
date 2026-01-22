from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
from parlai.core.metrics import (
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save
def update_history(self, obs: Message, temp_history: Optional[str]=None):
    """
        Update the history with the given observation.

        :param obs:
            Observation used to update the history.
        :param temp_history:
            Optional temporary string. If it is not None, this string will be
            appended to the end of the history. It will not be in the history
            on the next dialogue turn. Set to None to stop adding to the
            history.
        """
    if self.field in obs and obs[self.field] is not None:
        if self.split_on_newln:
            next_texts = obs[self.field].split('\n')
        else:
            next_texts = [obs[self.field]]
        for text in next_texts:
            self._update_raw_strings(text)
            if self.add_person_tokens:
                text = self._add_person_tokens(obs[self.field], self.p1_token, self.add_p1_after_newln)
            self._update_strings(text)
            self._update_vecs(text)
    self.temp_history = temp_history