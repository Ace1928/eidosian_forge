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
@pytest.mark.issue(4030)
def test_issue4030():
    """Test whether textcat works fine with empty doc"""
    unique_classes = ['offensive', 'inoffensive']
    x_train = ['This is an offensive text', 'This is the second offensive text', 'inoff']
    y_train = ['offensive', 'offensive', 'inoffensive']
    nlp = spacy.blank('en')
    train_data = []
    for text, train_instance in zip(x_train, y_train):
        cat_dict = {label: label == train_instance for label in unique_classes}
        train_data.append(Example.from_dict(nlp.make_doc(text), {'cats': cat_dict}))
    model = {'@architectures': 'spacy.TextCatBOW.v1', 'exclusive_classes': True, 'ngram_size': 2, 'no_output_layer': False}
    textcat = nlp.add_pipe('textcat', config={'model': model}, last=True)
    for label in unique_classes:
        textcat.add_label(label)
    with nlp.select_pipes(enable='textcat'):
        optimizer = nlp.initialize()
        for i in range(3):
            losses = {}
            batches = util.minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(examples=batch, sgd=optimizer, drop=0.1, losses=losses)
    doc = nlp('')
    assert doc.cats['offensive'] == 0.0
    assert doc.cats['inoffensive'] == 0.0