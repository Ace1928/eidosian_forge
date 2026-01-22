import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(4373)
def test_issue4373():
    """Test that PhraseMatcher.vocab can be accessed (like Matcher.vocab)."""
    matcher = Matcher(Vocab())
    assert isinstance(matcher.vocab, Vocab)
    matcher = PhraseMatcher(Vocab())
    assert isinstance(matcher.vocab, Vocab)