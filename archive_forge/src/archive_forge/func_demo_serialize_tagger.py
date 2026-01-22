import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_serialize_tagger():
    """
    Serializes the learned tagger to a file in pickle format; reloads it
    and validates the process.
    """
    postag(serialize_output='tagger.pcl')