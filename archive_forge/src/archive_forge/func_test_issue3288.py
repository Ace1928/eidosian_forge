import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(3288)
def test_issue3288(en_vocab):
    """Test that retokenization works correctly via displaCy when punctuation
    is merged onto the preceeding token and tensor is resized."""
    words = ['Hello', 'World', '!', 'When', 'is', 'this', 'breaking', '?']
    heads = [1, 1, 1, 4, 4, 6, 4, 4]
    deps = ['intj', 'ROOT', 'punct', 'advmod', 'ROOT', 'det', 'nsubj', 'punct']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    doc.tensor = numpy.zeros((len(words), 96), dtype='float32')
    displacy.render(doc)