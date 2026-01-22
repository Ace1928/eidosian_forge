import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_parse_ents_with_kb_id_options(en_vocab):
    """Test that named entities with kb_id on a Doc are converted into displaCy's format."""
    doc = Doc(en_vocab, words=['But', 'Google', 'is', 'starting', 'from', 'behind'])
    doc.ents = [Span(doc, 1, 2, label=doc.vocab.strings['ORG'], kb_id='Q95')]
    ents = displacy.parse_ents(doc, {'kb_url_template': 'https://www.wikidata.org/wiki/{}'})
    assert isinstance(ents, dict)
    assert ents['text'] == 'But Google is starting from behind '
    assert ents['ents'] == [{'start': 4, 'end': 10, 'label': 'ORG', 'kb_id': 'Q95', 'kb_url': 'https://www.wikidata.org/wiki/Q95'}]