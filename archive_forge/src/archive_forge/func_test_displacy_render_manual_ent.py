import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_render_manual_ent():
    """Test displacy.render with manual data for ent style"""
    parsed_ents = [{'text': 'But Google is starting from behind.', 'ents': [{'start': 4, 'end': 10, 'label': 'ORG'}]}, {'text': 'But Google is starting from behind.', 'ents': [{'start': -100, 'end': 100, 'label': 'COMPANY'}], 'title': 'Title'}]
    html = displacy.render(parsed_ents, style='ent', manual=True)
    for parsed_ent in parsed_ents:
        assert parsed_ent['ents'][0]['label'] in html
        if 'title' in parsed_ent:
            assert parsed_ent['title'] in html