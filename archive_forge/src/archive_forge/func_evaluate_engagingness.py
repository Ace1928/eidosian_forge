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
def evaluate_engagingness(self):
    control_msg = self.get_control_msg()
    msg_rng = len(ENGAGINGNESS_MSGS)
    for i in range(msg_rng):
        control_msg['text'] = ENGAGINGNESS_MSGS[i]
        control_msg['button_choices'] = '</ROUND>'.join(ENGAGINGNESS_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = ENGAGINGNESS_CHOICES.index(act.get('text'))
        self.engagingness_scores.append(act_choice)
    return True