import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(2728)
def test_issue2728(en_vocab):
    """Test that displaCy ENT visualizer escapes HTML correctly."""
    doc = Doc(en_vocab, words=['test', '<RELEASE>', 'test'])
    doc.ents = [Span(doc, 0, 1, label='TEST')]
    html = displacy.render(doc, style='ent')
    assert '&lt;RELEASE&gt;' in html
    doc.ents = [Span(doc, 1, 2, label='TEST')]
    html = displacy.render(doc, style='ent')
    assert '&lt;RELEASE&gt;' in html