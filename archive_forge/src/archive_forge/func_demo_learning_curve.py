import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_learning_curve():
    """
    Plot a learning curve -- the contribution on tagging accuracy of
    the individual rules.
    Note: requires matplotlib
    """
    postag(incremental_stats=True, separate_baseline_data=True, learning_curve_output='learningcurve.png')