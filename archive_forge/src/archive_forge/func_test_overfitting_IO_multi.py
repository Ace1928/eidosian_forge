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
def test_overfitting_IO_multi():
    fix_random_seed(0)
    nlp = English()
    textcat = nlp.add_pipe('textcat_multilabel')
    train_examples = []
    for text, annotations in TRAIN_DATA_MULTI_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    assert textcat.model.get_dim('nO') == 3
    for i in range(100):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['textcat_multilabel'] < 0.01
    test_text = 'I am confused but happy.'
    doc = nlp(test_text)
    cats = doc.cats
    assert cats['HAPPY'] > 0.9
    assert cats['CONFUSED'] > 0.9
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        cats2 = doc2.cats
        assert cats2['HAPPY'] > 0.9
        assert cats2['CONFUSED'] > 0.9
    scores = nlp.evaluate(train_examples)
    assert scores['cats_micro_f'] == 1.0
    assert scores['cats_macro_f'] == 1.0
    assert 'cats_score_desc' in scores
    texts = ['Just a sentence.', 'I like green eggs.', 'I am happy.', 'I eat ham.']
    batch_deps_1 = [doc.cats for doc in nlp.pipe(texts)]
    batch_deps_2 = [doc.cats for doc in nlp.pipe(texts)]
    no_batch_deps = [doc.cats for doc in [nlp(text) for text in texts]]
    for cats_1, cats_2 in zip(batch_deps_1, batch_deps_2):
        for cat in cats_1:
            assert_almost_equal(cats_1[cat], cats_2[cat], decimal=5)
    for cats_1, cats_2 in zip(batch_deps_1, no_batch_deps):
        for cat in cats_1:
            assert_almost_equal(cats_1[cat], cats_2[cat], decimal=5)