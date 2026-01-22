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
def evaluate_consistency(self):
    control_msg = self.get_control_msg()
    control_msg['text'] = CONSISTENCY_MSGS[0]
    control_msg['button_choices'] = '</ROUND>'.join(CONSISTENCY_CHOICES)
    self.eval_agent.observe(validate(control_msg))
    act = self.eval_agent.act(timeout=self.max_resp_time)
    timeout = self.check_timeout(act)
    if timeout:
        return False
    act_choice = CONSISTENCY_CHOICES.index(act.get('text'))
    self.consistency_scores.append(act_choice)
    if ASK_DETAILED and act_choice != 0:
        control_msg = self.get_control_msg()
        control_msg['text'] = CONSISTENCY_MSGS[1]
        control_msg['good_rounds'] = True
        control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        if 'text' in act:
            self.consistency_scores.append([int(x) - 1 for x in act['text'].split(',')])
    return True