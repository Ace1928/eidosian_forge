import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def extrep_ngram_used_before(dict, hypothesis, history, wt, feat, n, person):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that, if added to hypothesis, would create a n-gram that has already been used
    earlier in the conversation; otherwise 0.

    Additional inputs:
      n: int, the size of the n-grams considered.
      person: If 'self', identify n-grams that have already been used by self (bot).
        If 'partner', identify n-grams that have already been used by partner (human).
    """
    if person == 'self':
        prev_utts = history.own_utts
    elif person == 'partner':
        prev_utts = history.partner_utts
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return feat
    if hypothesis is None:
        return feat
    prev_utts_wordidx = [dict.txt2vec(utt) for utt in prev_utts]
    bad_words = [matching_ngram_completions(prev_utt, hypothesis, n) for prev_utt in prev_utts_wordidx]
    bad_words = list(set(flatten(bad_words)))
    if len(bad_words) > 0:
        feat[bad_words] += wt
    return feat