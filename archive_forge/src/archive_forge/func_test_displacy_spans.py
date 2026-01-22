import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_spans(en_vocab):
    """Test that displaCy can render Spans."""
    doc = Doc(en_vocab, words=['But', 'Google', 'is', 'starting', 'from', 'behind'])
    doc.ents = [Span(doc, 1, 2, label=doc.vocab.strings['ORG'])]
    html = displacy.render(doc[1:4], style='ent')
    assert html.startswith('<div')