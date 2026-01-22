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
@pytest.mark.issue(4849)
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_issue4849(entity_ruler_factory):
    nlp = English()
    patterns = [{'label': 'PERSON', 'pattern': 'joe biden', 'id': 'joe-biden'}, {'label': 'PERSON', 'pattern': 'bernie sanders', 'id': 'bernie-sanders'}]
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler', config={'phrase_matcher_attr': 'LOWER'})
    ruler.add_patterns(patterns)
    text = '\n    The left is starting to take aim at Democratic front-runner Joe Biden.\n    Sen. Bernie Sanders joined in her criticism: "There is no \'middle ground\' when it comes to climate policy."\n    '
    count_ents = 0
    for doc in nlp.pipe([text], n_process=1):
        count_ents += len([ent for ent in doc.ents if ent.ent_id > 0])
    assert count_ents == 2
    if isinstance(get_current_ops, NumpyOps):
        count_ents = 0
        for doc in nlp.pipe([text], n_process=2):
            count_ents += len([ent for ent in doc.ents if ent.ent_id > 0])
        assert count_ents == 2