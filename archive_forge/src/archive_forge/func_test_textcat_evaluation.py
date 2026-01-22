import random
import numpy.random
import pytest
from numpy.testing import assert_almost_equal
from thinc.api import Config, compounding, fix_random_seed, get_current_ops
from wasabi import msg
import spacy
from spacy import util
from spacy.cli.evaluate import print_prf_per_type, print_textcats_auc_per_cat
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import TextCategorizer
from spacy.pipeline.textcat import (
from spacy.pipeline.textcat_multilabel import (
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.training.initialize import init_nlp
from ..tok2vec import build_lazy_init_tok2vec as _
from ..util import make_tempdir
def test_textcat_evaluation():
    train_examples = []
    nlp = English()
    ref1 = nlp('one')
    ref1.cats = {'winter': 1.0, 'summer': 1.0, 'spring': 1.0, 'autumn': 1.0}
    pred1 = nlp('one')
    pred1.cats = {'winter': 1.0, 'summer': 0.0, 'spring': 1.0, 'autumn': 1.0}
    train_examples.append(Example(pred1, ref1))
    ref2 = nlp('two')
    ref2.cats = {'winter': 0.0, 'summer': 0.0, 'spring': 1.0, 'autumn': 1.0}
    pred2 = nlp('two')
    pred2.cats = {'winter': 1.0, 'summer': 0.0, 'spring': 0.0, 'autumn': 1.0}
    train_examples.append(Example(pred2, ref2))
    scores = Scorer().score_cats(train_examples, 'cats', labels=['winter', 'summer', 'spring', 'autumn'])
    assert scores['cats_f_per_type']['winter']['p'] == 1 / 2
    assert scores['cats_f_per_type']['winter']['r'] == 1 / 1
    assert scores['cats_f_per_type']['summer']['p'] == 0
    assert scores['cats_f_per_type']['summer']['r'] == 0 / 1
    assert scores['cats_f_per_type']['spring']['p'] == 1 / 1
    assert scores['cats_f_per_type']['spring']['r'] == 1 / 2
    assert scores['cats_f_per_type']['autumn']['p'] == 2 / 2
    assert scores['cats_f_per_type']['autumn']['r'] == 2 / 2
    assert scores['cats_micro_p'] == 4 / 5
    assert scores['cats_micro_r'] == 4 / 6