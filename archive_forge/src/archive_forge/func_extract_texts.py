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
def extract_texts(self, obs):
    """
        Extract the personalities and dialogue histories from observations.

        Additionally determine which dialogue round we are in.

        Note that this function assumes that the personality is the
        last line of the `text` field in the observation.

        :param obs:
            list of observations

        :return:
            a list of personalities, a list of dialogue histories, and the
            current dialogue round (either first, second, or third+)
        """
    splits = [v.get('text').split('\n') for v in obs]
    if self.personality_override:
        splits = [s + [self.personality_override] for s in splits]
    personalities = [t[-1] for t in splits]
    dialogue_histories = None
    dialogue_round = 'first_round'
    if len(splits[0]) >= 2:
        dialogue_round = 'second_round' if len(splits[0]) == 2 else 'third_round+'
        dialogue_histories = ['\n'.join(t[:-1]) for t in splits]
    return (personalities, dialogue_histories, dialogue_round)