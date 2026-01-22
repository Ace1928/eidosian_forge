from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
from extract_and_save_personas import main as main_extract
import numpy as np
import time
import os
import pickle
import random
def is_exact_match(self, act, ag, tolerance=0):
    if act['episode_done']:
        return False
    control_msg = {'episode_done': False}
    control_msg['id'] = 'SYSTEM'
    text = act['text']
    if text not in ['', ' ', '  ', '   ']:
        n_word_match = 0
        for per in ag.persona_data:
            per_parse = per.split(' ')
            regular_words = ['', ' ', 'I', "I'm", 'My', 'i']
            for r_w in regular_words:
                if r_w in per_parse:
                    per_parse.remove(r_w)
            per_subseq = [' '.join(per_parse[i:i + len(per_parse) - tolerance]) for i in range(tolerance + 1)]
            for pp in per_subseq:
                if pp in ['', ' ', '  ', '   ']:
                    per_subseq.remove(pp)
            n_word_match += sum([paa in text for paa in per_subseq])
        if n_word_match > 0:
            control_msg['text'] = 'We found that you <b><span style="color:red">trivially copied character descriptions</span></b>. Please rephrase your message again.'
            ag.observe(validate(control_msg))
            return True
        else:
            return False