import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_generated_templates():
    """
    Template.expand and Feature.expand are class methods facilitating
    generating large amounts of templates. See their documentation for
    details.

    Note: training with 500 templates can easily fill all available
    even on relatively small corpora
    """
    wordtpls = Word.expand([-1, 0, 1], [1, 2], excludezero=False)
    tagtpls = Pos.expand([-2, -1, 0, 1], [1, 2], excludezero=True)
    templates = list(Template.expand([wordtpls, tagtpls], combinations=(1, 3)))
    print('Generated {} templates for transformation-based learning'.format(len(templates)))
    postag(templates=templates, incremental_stats=True, template_stats=True)