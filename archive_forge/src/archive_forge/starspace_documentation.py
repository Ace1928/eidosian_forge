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

        Return opt and model states.
        