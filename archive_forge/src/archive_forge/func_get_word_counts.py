from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.build_data import modelzoo_path
import torchtext.vocab as vocab
from parlai.utils.misc import TimeLogger
from collections import Counter, deque
import numpy as np
import os
import pickle
import torch
def get_word_counts(opt, count_inputs):
    """
    Goes through the dataset specified in opt, returns word counts and all utterances.

    Inputs:
      count_inputs: If True, include both input and reply when counting words and
        utterances. Otherwise, only include reply text.

    Returns:
      word_counter: a Counter mapping each word to the total number of times it appears
      total_count: int. total word count, i.e. the sum of the counts for each word
      all_utts: list of strings. all the utterances that were used for counting words
    """
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    word_counter = Counter()
    total_count = 0
    all_utts = []
    log_timer = TimeLogger()
    while True:
        world.parley()
        reply = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0]
        words = reply.split()
        word_counter.update(words)
        total_count += len(words)
        all_utts.append(reply)
        if count_inputs:
            input = world.acts[0]['text']
            input = input.split('\n')[-1]
            words = input.split()
            word_counter.update(words)
            total_count += len(words)
            all_utts.append(input)
        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            print(text)
        if world.epoch_done():
            print('EPOCH DONE')
            break
    assert total_count == sum(word_counter.values())
    return (word_counter, total_count, all_utts)