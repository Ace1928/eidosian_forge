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
@pytest.mark.parametrize('component_name,scorer', [('textcat', 'spacy.textcat_scorer.v1'), ('textcat_multilabel', 'spacy.textcat_multilabel_scorer.v1')])
def test_textcat_legacy_scorers(component_name, scorer):
    """Check that legacy scorers are registered and produce the expected score
    keys."""
    nlp = English()
    nlp.add_pipe(component_name, config={'scorer': {'@scorers': scorer}})
    train_examples = []
    for text, annotations in TRAIN_DATA_SINGLE_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    nlp.initialize(get_examples=lambda: train_examples)
    scores = nlp.evaluate(train_examples)
    assert 0 <= scores['cats_score'] <= 1