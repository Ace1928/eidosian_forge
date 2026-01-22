import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def extrep_repeated_word_frac(utt, history, remove_stopwords, person):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of words in utt that already appeared in a previous utterance.
    Additional inputs:
      remove_stopwords: bool. If True, stopwords are removed from utt before counting
        repetition.
      person: If 'self', identify words that have already been used by self (bot).
        If 'partner', identify words that have already been used by partner (human).
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
    tokens = utt.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    prev_words = [s.split() for s in prev_utts]
    prev_words = list(set(flatten(prev_words)))
    return extrep_frac(tokens, prev_words)