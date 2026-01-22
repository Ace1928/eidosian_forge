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
def get_human_agent_act(self, agent):
    act = agent.act(timeout=self.max_resp_time)
    while self.is_msg_tooshortlong(act, agent):
        act = agent.act(timeout=self.max_resp_time)
    return act