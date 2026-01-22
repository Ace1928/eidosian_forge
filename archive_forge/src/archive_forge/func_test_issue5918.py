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
@pytest.mark.issue(5918)
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_issue5918(entity_ruler_factory):
    nlp = English()
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'ORG', 'pattern': 'Digicon Inc'}, {'label': 'ORG', 'pattern': "Rotan Mosle Inc's"}, {'label': 'ORG', 'pattern': 'Rotan Mosle Technology Partners Ltd'}]
    ruler.add_patterns(patterns)
    text = "\n        Digicon Inc said it has completed the previously-announced disposition\n        of its computer systems division to an investment group led by\n        Rotan Mosle Inc's Rotan Mosle Technology Partners Ltd affiliate.\n        "
    doc = nlp(text)
    assert len(doc.ents) == 3
    doc = merge_entities(doc)
    assert len(doc.ents) == 3