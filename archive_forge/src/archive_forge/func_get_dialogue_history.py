from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from parlai.core.message import Message
from .modules import TransresnetMultimodalModel
from projects.personality_captions.transresnet.transresnet import TransresnetAgent
import torch
from torch import optim
import random
import os
import numpy as np
import tqdm
from collections import deque
def get_dialogue_history(self, obs):
    """
        Get dialogue history for an observation.

        :param obs:
            observation

        :return:
            the observation with the dialogue history in the `text` field
        """
    if len(self.history) > 0:
        obs.force_set('text', '\n'.join(self.history) + '\n' + obs['text'])
    if 'labels' in obs:
        self.history.append(random.choice(obs['labels']))
    elif 'eval_labels' in obs:
        self.history.append(random.choice(obs['eval_labels']))
    if obs.get('episode_done', True):
        self.history.clear()
    return obs