import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(4651)
def test_issue4651_with_phrase_matcher_attr():
    """Test that the EntityRuler PhraseMatcher is deserialized correctly using
    the method from_disk when the EntityRuler argument phrase_matcher_attr is
    specified.
    """
    text = 'Spacy is a python library for nlp'
    nlp = English()
    patterns = [{'label': 'PYTHON_LIB', 'pattern': 'spacy', 'id': 'spaCy'}]
    ruler = nlp.add_pipe('entity_ruler', config={'phrase_matcher_attr': 'LOWER'})
    ruler.add_patterns(patterns)
    doc = nlp(text)
    res = [(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents]
    nlp_reloaded = English()
    with make_tempdir() as d:
        file_path = d / 'entityruler'
        ruler.to_disk(file_path)
        nlp_reloaded.add_pipe('entity_ruler').from_disk(file_path)
    doc_reloaded = nlp_reloaded(text)
    res_reloaded = [(ent.text, ent.label_, ent.ent_id_) for ent in doc_reloaded.ents]
    assert res == res_reloaded