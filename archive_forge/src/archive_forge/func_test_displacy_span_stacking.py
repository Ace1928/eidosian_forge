import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(13056)
def test_displacy_span_stacking():
    """Test whether span stacking works properly for multiple overlapping spans."""
    spans = [{'start_token': 2, 'end_token': 5, 'label': 'SkillNC'}, {'start_token': 0, 'end_token': 2, 'label': 'Skill'}, {'start_token': 1, 'end_token': 3, 'label': 'Skill'}]
    tokens = ['Welcome', 'to', 'the', 'Bank', 'of', 'China', '.']
    per_token_info = SpanRenderer._assemble_per_token_info(spans=spans, tokens=tokens)
    assert len(per_token_info) == len(tokens)
    assert all([len(per_token_info[i]['entities']) == 1 for i in (0, 3, 4)])
    assert all([len(per_token_info[i]['entities']) == 2 for i in (1, 2)])
    assert per_token_info[1]['entities'][0]['render_slot'] == 1
    assert per_token_info[1]['entities'][1]['render_slot'] == 2
    assert per_token_info[2]['entities'][0]['render_slot'] == 2
    assert per_token_info[2]['entities'][1]['render_slot'] == 3