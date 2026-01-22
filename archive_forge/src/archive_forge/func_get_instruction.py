from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.legacy_2018.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.legacy_2018.worlds import MTurkOnboardWorld
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply
from joblib import Parallel, delayed
import numpy as np
import os
import json
import random
import time
import torch
import copy
def get_instruction(self, agent_id=None, tag='first'):
    if tag == 'start':
        return START_MSG.format(self.n_turn)
    if tag == 'chat_not_done':
        return CHAT_NOT_DONE_MSG.format(self.n_turn + 1 - self.turn_idx)
    if tag == 'timeout':
        return TIMEOUT_MESSAGE
    if tag == 'exceed_min_turns':
        return EXCEED_MIN_TURNS_MSG.format(self.n_turn)