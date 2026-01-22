from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.core.worlds import validate
from parlai.mturk.tasks.personachat.personachat_chat.extract_and_save_personas import (
from joblib import Parallel, delayed
import numpy as np
import time
import os
import pickle
import random
def is_close_match(self, act, ag, persona_data, tolerance=0.7):
    if act['episode_done']:
        return False
    control_msg = {'episode_done': False}
    control_msg['id'] = 'SYSTEM'
    text = act['text']
    if text not in ['', ' ', '  ', '   ']:
        n_word_match = 0
        per_parse = persona_data.split(' ')
        regular_words = ['', ' ', 'I', "I'm", 'My', 'i']
        for r_w in regular_words:
            if r_w in per_parse:
                per_parse.remove(r_w)
        n_word_match += sum([word in text for word in per_parse])
        if n_word_match / (len(per_parse) + 1) > tolerance:
            control_msg['text'] = 'We found that you <b><span style="color:red">trivially copied character descriptions</span></b>. Please rephrase your message again.'
            ag.observe(validate(control_msg))
            return True
        else:
            return False