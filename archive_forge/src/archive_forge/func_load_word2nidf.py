from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from collections import Counter
import os
import math
import pickle
def load_word2nidf(opt):
    """
    Loads word count stats from word2count.pkl file in data/controllable_dialogue,
    computes NIDF for all words, and returns the word2nidf dictionary.

    Returns:
      word2nidf: dict mapping words to their NIDF score (float between 0 and 1)
    """
    word2count_fp = os.path.join(opt['datapath'], CONTROLLABLE_DIR, 'word2count.pkl')
    print('Loading word count stats from %s...' % word2count_fp)
    with open(word2count_fp, 'rb') as f:
        data = pickle.load(f)
    num_sents = data['num_sents']
    print('num_sents: ', num_sents)
    word2count = data['word2count']
    min_c = min(word2count.values())
    max_c = max(word2count.values())
    word2nidf = {w: (math.log(max_c) - math.log(c)) / (math.log(max_c) - math.log(min_c)) for w, c in word2count.items()}
    print('Done loading word2nidf dictionary.')
    return word2nidf