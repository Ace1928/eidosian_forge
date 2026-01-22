import math
from collections.abc import Sequence
import heapq
import json
import torch
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
def score_match(query_rep, text, length_penalty, dictionary=None, debug=False):
    """
    Calculate the score match between the query representation the text.

    :param query_rep: base query representation to match text again.
    :param text: string to comapre against query_rep for matching tokens
    :param length_penalty: scores are divided by the norm taken to this power
    :dictionary: optional dictionary to use to tokenize text
    :debug: flag to enable printing every match

    :returns: float score of match
    """
    if text == '':
        return 0
    if not dictionary:
        words = text.lower().split(' ')
    else:
        words = [w for w in dictionary.tokenize(text.lower())]
    score = 0
    rw = query_rep['words']
    used = {}
    for w in words:
        if w in rw and w not in used:
            score += rw[w]
            if debug:
                print('match: ' + w)
        used[w] = True
    norm = math.sqrt(len(used))
    norm = math.pow(norm * query_rep['norm'], length_penalty)
    if norm > 1:
        score /= norm
    return score