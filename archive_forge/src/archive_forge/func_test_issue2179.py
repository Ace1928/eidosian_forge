import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(2179)
def test_issue2179():
    """Test that spurious 'extra_labels' aren't created when initializing NER."""
    nlp = Italian()
    ner = nlp.add_pipe('ner')
    ner.add_label('CITIZENSHIP')
    nlp.initialize()
    nlp2 = Italian()
    nlp2.add_pipe('ner')
    assert len(nlp2.get_pipe('ner').labels) == 0
    model = nlp2.get_pipe('ner').model
    model.attrs['resize_output'](model, nlp.get_pipe('ner').moves.n_moves)
    nlp2.from_bytes(nlp.to_bytes())
    assert 'extra_labels' not in nlp2.get_pipe('ner').cfg
    assert nlp2.get_pipe('ner').labels == ('CITIZENSHIP',)