import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_high_accuracy_rules():
    """
    Discard rules with low accuracy. This may hurt performance a bit,
    but will often produce rules which are more interesting read to a human.
    """
    postag(num_sents=3000, min_acc=0.96, min_score=10)