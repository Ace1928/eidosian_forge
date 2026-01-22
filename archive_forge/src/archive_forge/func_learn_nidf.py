from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from collections import Counter
import os
import math
import pickle
def learn_nidf(opt):
    """
    Go through ConvAI2 and Twitter data, and count word frequences.

    Save word2count.pkl, which contains word2count, and total num_sents. These are both
    needed to calculate NIDF later.
    """
    opt['log_every_n_secs'] = 2
    print('Counting words in Twitter train set...')
    opt['datatype'] = 'train:ordered'
    opt['task'] = 'twitter'
    wc1, ns1 = get_word_counts(opt, count_inputs=True)
    print('Counting words in Twitter val set...')
    opt['datatype'] = 'valid'
    opt['task'] = 'twitter'
    wc2, ns2 = get_word_counts(opt, count_inputs=True)
    opt['task'] = 'fromfile:parlaiformat'
    print('Counting words in ConvAI2 train set...')
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = os.path.join(opt['datapath'], PARLAI_FORMAT_DIR, 'train.txt')
    wc3, ns3 = get_word_counts(opt, count_inputs=False)
    print('Counting words in ConvAI2 val set...')
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = os.path.join(opt['datapath'], PARLAI_FORMAT_DIR, 'valid.txt')
    wc4, ns4 = get_word_counts(opt, count_inputs=True)
    word_counter = Counter()
    for wc in [wc1, wc2, wc3, wc4]:
        for word, count in wc.items():
            word_counter[word] += count
    num_sents = ns1 + ns2 + ns3 + ns4
    word2count_fp = os.path.join(opt['datapath'], CONTROLLABLE_DIR, 'word2count.pkl')
    print('Saving word count stats to %s...' % word2count_fp)
    data = {'word2count': word_counter, 'num_sents': num_sents}
    with open(word2count_fp, 'wb') as f:
        pickle.dump(data, f)