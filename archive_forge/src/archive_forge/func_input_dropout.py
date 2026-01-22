from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import maintain_dialog_history, load_cands
from parlai.core.torch_agent import TorchAgent
from .modules import Starspace
import torch
from torch import optim
import torch.nn as nn
from collections import deque
import copy
import os
import random
import json
def input_dropout(self, xs, ys, negs):

    def dropout(x, rate):
        xd = []
        for i in x[0]:
            if random.uniform(0, 1) > rate:
                xd.append(i)
        if len(xd) == 0:
            xd.append(x[0][random.randint(0, x.size(1) - 1)])
        return torch.LongTensor(xd).unsqueeze(0)
    rate = self.opt.get('input_dropout')
    xs2 = dropout(xs, rate)
    ys2 = dropout(ys, rate)
    negs2 = []
    for n in negs:
        negs2.append(dropout(n, rate))
    return (xs2, ys2, negs2)