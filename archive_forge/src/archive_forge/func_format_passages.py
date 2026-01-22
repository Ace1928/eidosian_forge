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
def format_passages(self, ir_passages, max_length=MAX_DOC_LEN):
    passages = []
    if len(ir_passages) == 1:
        passages.append(['No Passages Retrieved', []])
    else:
        for passage in ir_passages:
            split = passage.split('\n')
            title = split[0]
            split = self.sent_tok.tokenize(' '.join(split[1:]))
            split[0] = split[0][1:]
            sentences = []
            for sent in split:
                if len(sent) > 1:
                    sentences.append(sent)
                    if len(' '.join(sentences)) > max_length:
                        break
            passages.append([title, sentences])
    return passages