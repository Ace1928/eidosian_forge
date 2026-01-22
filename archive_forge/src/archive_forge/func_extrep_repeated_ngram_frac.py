import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def extrep_repeated_ngram_frac(utt, history, n, person):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns fraction of n-grams in utt that already appeared in a previous utterance.
    Additional inputs:
      n: int, the size of the n-grams considered.
      person: If 'self', identify n-grams that have already been used by self (bot).
        If 'partner', identify n-grams that have already been used by partner (human).
    """
    assert utt.strip() != ''
    if person == 'self':
        prev_utts = history.own_utts
    elif person == 'partner':
        prev_utts = history.partner_utts
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return 0
    utt_ngrams = get_ngrams(utt, n)
    prev_ngrams = [get_ngrams(prev, n) for prev in prev_utts]
    prev_ngrams = list(set(flatten(prev_ngrams)))
    return extrep_frac(utt_ngrams, prev_ngrams)