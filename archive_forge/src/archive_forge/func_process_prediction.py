from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from controllable_seq2seq.controls import (
from controllable_seq2seq.util import ConvAI2History
from collections import Counter
import copy
import random
import json
import time
import os
def process_prediction(prediction, word_statistics):
    word_statistics['pred_list'].append(normalize_answer(prediction))
    freqs, _cnt, wlength, clength = get_word_stats(prediction, dictionary, bins=bins)
    word_statistics['word_cnt'] += _cnt
    word_statistics['mean_wlength'].append(wlength)
    word_statistics['mean_clength'].append(clength)
    word_statistics['freqs_cnt'] += Counter(freqs)
    return word_statistics