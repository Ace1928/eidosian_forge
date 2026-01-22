import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def avg_nidf(utt, history):
    """
    Sentence-level attribute function.

    See explanation above. Returns the mean NIDF of the words in utt.
    """
    words = utt.split()
    problem_words = [w for w in words if w not in word2nidf]
    ok_words = [w for w in words if w in word2nidf]
    if len(ok_words) == 0:
        print("WARNING: For all the words in the utterance '%s', we do not have the NIDF score. Marking as avg_nidf=1." % utt)
        return 1
    nidfs = [word2nidf[w] for w in ok_words]
    avg_nidf = sum(nidfs) / len(nidfs)
    if len(problem_words) > 0:
        print("WARNING: When calculating avg_nidf for the utterance '%s', we don't know NIDF for the following words: %s" % (utt, str(problem_words)))
    assert avg_nidf >= 0 and avg_nidf <= 1
    return avg_nidf