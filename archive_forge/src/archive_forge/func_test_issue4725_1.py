import pickle
import pytest
import srsly
from thinc.api import Linear
import spacy
from spacy import Vocab, load, registry
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import (
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.tokens import Span
from spacy.util import ensure_path, load_model
from ..util import make_tempdir
@pytest.mark.issue(4725)
def test_issue4725_1():
    """Ensure the pickling of the NER goes well"""
    vocab = Vocab(vectors_name='test_vocab_add_vector')
    nlp = English(vocab=vocab)
    config = {'update_with_oracle_cut_size': 111}
    ner = nlp.create_pipe('ner', config=config)
    with make_tempdir() as tmp_path:
        with (tmp_path / 'ner.pkl').open('wb') as file_:
            pickle.dump(ner, file_)
            assert ner.cfg['update_with_oracle_cut_size'] == 111
        with (tmp_path / 'ner.pkl').open('rb') as file_:
            ner2 = pickle.load(file_)
            assert ner2.cfg['update_with_oracle_cut_size'] == 111