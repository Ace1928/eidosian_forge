import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
def space_punctuation(self, words, unspace=False):
    puncs = [('.', ' .'), (',', ' ,'), ('?', ' ?'), (' !', '!'), ('(', ' ('), (')', ' )')]
    new_words = words
    for punc in puncs:
        if unspace:
            new_words = new_words.replace(punc[1], punc[0])
        else:
            new_words = new_words.replace(punc[0], punc[1])
    return new_words