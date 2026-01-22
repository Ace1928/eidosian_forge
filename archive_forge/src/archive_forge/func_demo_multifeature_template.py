import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_multifeature_template():
    """
    Templates can have more than a single feature.
    """
    postag(templates=[Template(Word([0]), Pos([-2, -1]))])