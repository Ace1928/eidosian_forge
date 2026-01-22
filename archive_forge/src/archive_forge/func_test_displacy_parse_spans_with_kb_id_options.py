import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_parse_spans_with_kb_id_options(en_vocab):
    """Test that spans with kb_id on a Doc are converted into displaCy's format"""
    doc = Doc(en_vocab, words=['Welcome', 'to', 'the', 'Bank', 'of', 'China'])
    doc.spans['sc'] = [Span(doc, 3, 6, 'ORG', kb_id='Q790068'), Span(doc, 5, 6, 'GPE', kb_id='Q148')]
    spans = displacy.parse_spans(doc, {'kb_url_template': 'https://wikidata.org/wiki/{}'})
    assert isinstance(spans, dict)
    assert spans['text'] == 'Welcome to the Bank of China '
    assert spans['spans'] == [{'start': 15, 'end': 28, 'start_token': 3, 'end_token': 6, 'label': 'ORG', 'kb_id': 'Q790068', 'kb_url': 'https://wikidata.org/wiki/Q790068'}, {'start': 23, 'end': 28, 'start_token': 5, 'end_token': 6, 'label': 'GPE', 'kb_id': 'Q148', 'kb_url': 'https://wikidata.org/wiki/Q148'}]