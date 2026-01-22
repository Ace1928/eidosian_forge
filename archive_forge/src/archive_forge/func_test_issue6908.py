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
@pytest.mark.parametrize('component_name', ['textcat', 'textcat_multilabel'])
@pytest.mark.issue(6908)
def test_issue6908(component_name):
    """Test intializing textcat with labels in a list"""

    def create_data(out_file):
        nlp = spacy.blank('en')
        doc = nlp.make_doc('Some text')
        doc.cats = {'label1': 0, 'label2': 1}
        out_data = DocBin(docs=[doc]).to_bytes()
        with out_file.open('wb') as file_:
            file_.write(out_data)
    with make_tempdir() as tmp_path:
        train_path = tmp_path / 'train.spacy'
        create_data(train_path)
        config_str = CONFIG_ISSUE_6908.replace('TEXTCAT_PLACEHOLDER', component_name)
        config_str = config_str.replace('TRAIN_PLACEHOLDER', train_path.as_posix())
        config = util.load_config_from_str(config_str)
        init_nlp(config)