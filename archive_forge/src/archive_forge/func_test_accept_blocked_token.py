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
def test_accept_blocked_token():
    """Test succesful blocking of tokens to be in an entity."""
    nlp1 = English()
    doc1 = nlp1('I live in New York')
    config = {}
    ner1 = nlp1.create_pipe('ner', config=config)
    assert [token.ent_iob_ for token in doc1] == ['', '', '', '', '']
    assert [token.ent_type_ for token in doc1] == ['', '', '', '', '']
    ner1.moves.add_action(5, '')
    ner1.add_label('GPE')
    state1 = ner1.moves.init_batch([doc1])[0]
    ner1.moves.apply_transition(state1, 'O')
    ner1.moves.apply_transition(state1, 'O')
    ner1.moves.apply_transition(state1, 'O')
    assert ner1.moves.is_valid(state1, 'B-GPE')
    nlp2 = English()
    doc2 = nlp2('I live in New York')
    config = {}
    ner2 = nlp2.create_pipe('ner', config=config)
    doc2.set_ents([], blocked=[doc2[3:5]], default='unmodified')
    assert [token.ent_iob_ for token in doc2] == ['', '', '', 'B', 'B']
    assert [token.ent_type_ for token in doc2] == ['', '', '', '', '']
    ner2.moves.add_action(4, '')
    ner2.moves.add_action(5, '')
    ner2.add_label('GPE')
    state2 = ner2.moves.init_batch([doc2])[0]
    ner2.moves.apply_transition(state2, 'O')
    ner2.moves.apply_transition(state2, 'O')
    ner2.moves.apply_transition(state2, 'O')
    assert not ner2.moves.is_valid(state2, 'B-GPE')
    assert ner2.moves.is_valid(state2, 'U-')
    ner2.moves.apply_transition(state2, 'U-')
    assert not ner2.moves.is_valid(state2, 'B-GPE')
    assert ner2.moves.is_valid(state2, 'U-')