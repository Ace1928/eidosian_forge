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
def match_batch(self, batch_reply, valid_inds, output=None):
    """
        Match sub-batch of predictions to the original batch indices.

        Batches may be only partially filled (i.e when completing the remainder
        at the end of the validation or test set), or we may want to sort by
        e.g the length of the input sequences if using pack_padded_sequence.

        This matches rows back with their original row in the batch for
        calculating metrics like accuracy.

        If output is None (model choosing not to provide any predictions), we
        will just return the batch of replies.

        Otherwise, output should be a parlai.core.torch_agent.Output object.
        This is a namedtuple, which can provide text predictions and/or
        text_candidates predictions. If you would like to map additional
        fields into the batch_reply, you can override this method as well as
        providing your own namedtuple with additional fields.

        :param batch_reply:
            Full-batchsize list of message dictionaries to put responses into.

        :param valid_inds:
            Original indices of the predictions.

        :param output:
            Output namedtuple which contains sub-batchsize list of text outputs
            from model. May be None (default) if model chooses not to answer.
            This method will check for ``text`` and ``text_candidates`` fields.
        """
    if output is None:
        return batch_reply
    for k, v in output.items():
        if v is None:
            continue
        for i, sub_val in zip(valid_inds, v):
            batch_reply[i][k] = sub_val
    return batch_reply