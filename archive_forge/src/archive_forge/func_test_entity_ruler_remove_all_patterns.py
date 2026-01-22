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
def test_entity_ruler_remove_all_patterns(nlp, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'PERSON', 'pattern': 'Dina', 'id': 'dina'}, {'label': 'ORG', 'pattern': 'ACME', 'id': 'acme'}, {'label': 'DATE', 'pattern': 'her birthday', 'id': 'bday'}]
    ruler.add_patterns(patterns)
    assert len(ruler.patterns) == 3
    if isinstance(ruler, EntityRuler):
        ruler.remove('dina')
    else:
        ruler.remove_by_id('dina')
    assert len(ruler.patterns) == 2
    if isinstance(ruler, EntityRuler):
        ruler.remove('acme')
    else:
        ruler.remove_by_id('acme')
    assert len(ruler.patterns) == 1
    if isinstance(ruler, EntityRuler):
        ruler.remove('bday')
    else:
        ruler.remove_by_id('bday')
    assert len(ruler.patterns) == 0
    with pytest.warns(UserWarning):
        doc = nlp('Dina founded her company ACME on her birthday')
        assert len(doc.ents) == 0