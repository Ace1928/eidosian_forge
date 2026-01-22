import pytest
from spacy.tokens import Doc
from spacy.util import filter_spans
@pytest.fixture
def nl_reference_chunking():
    return ['haar vriend', 'we', 'ruzie', 'we', 'de supermarkt', 'het begin', 'de supermarkt', 'het fruit', 'de groentes', 'we', 'geen avondeten']