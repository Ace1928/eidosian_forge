from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.utils.safety import OffensiveStringMatcher
from joblib import Parallel, delayed
from task_config import task_config as config
from extract_and_save_personas import main as main_extract
from constants import (
import numpy as np
import time
import os
import pickle
import random
import copy
from urllib.parse import unquote
def retrieve_passages(self, act, num_passages=None):
    if not num_passages:
        num_passages = self.num_passages_to_retrieve
    self.ir_agent.observe(act)
    action = self.ir_agent.act()
    passages = action.get('text_candidates', [action.get('text', '')])
    return passages[:min(len(passages), num_passages)]