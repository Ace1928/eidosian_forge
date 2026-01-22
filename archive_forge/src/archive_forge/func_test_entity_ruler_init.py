import pytest
from thinc.api import NumpyOps, get_current_ops
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import EntityRecognizer, EntityRuler, SpanRuler, merge_entities
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc, Span
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_entity_ruler_init(nlp, patterns, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    ruler.add_patterns(patterns)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    assert 'HELLO' in ruler
    assert 'BYE' in ruler
    nlp.remove_pipe('entity_ruler')
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    ruler.add_patterns(patterns)
    doc = nlp('hello world bye bye')
    assert len(doc.ents) == 2
    assert doc.ents[0].label_ == 'HELLO'
    assert doc.ents[1].label_ == 'BYE'