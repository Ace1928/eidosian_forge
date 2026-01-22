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
@pytest.mark.issue(7019)
def test_issue7019():
    scores = {'LABEL_A': 0.39829102, 'LABEL_B': 0.938298329382, 'LABEL_C': None}
    print_textcats_auc_per_cat(msg, scores)
    scores = {'LABEL_A': {'p': 0.3420302, 'r': 0.392902, 'f': 0.49823928932}, 'LABEL_B': {'p': None, 'r': None, 'f': None}}
    print_prf_per_type(msg, scores, name='foo', type='bar')