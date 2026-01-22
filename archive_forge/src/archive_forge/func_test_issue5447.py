import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(5447)
def test_issue5447():
    """Test that overlapping arcs get separate levels, unless they're identical."""
    renderer = DependencyRenderer()
    words = [{'text': 'This', 'tag': 'DT'}, {'text': 'is', 'tag': 'VBZ'}, {'text': 'a', 'tag': 'DT'}, {'text': 'sentence.', 'tag': 'NN'}]
    arcs = [{'start': 0, 'end': 1, 'label': 'nsubj', 'dir': 'left'}, {'start': 2, 'end': 3, 'label': 'det', 'dir': 'left'}, {'start': 2, 'end': 3, 'label': 'overlap', 'dir': 'left'}, {'end': 3, 'label': 'overlap', 'start': 2, 'dir': 'left'}, {'start': 1, 'end': 3, 'label': 'attr', 'dir': 'left'}]
    renderer.render([{'words': words, 'arcs': arcs}])
    assert renderer.highest_level == 3