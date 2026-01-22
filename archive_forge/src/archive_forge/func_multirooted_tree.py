import pytest
from spacy.pipeline._parser_internals import nonproj
from spacy.pipeline._parser_internals.nonproj import (
from spacy.tokens import Doc
@pytest.fixture
def multirooted_tree():
    return [3, 2, 0, 3, 3, 7, 7, 3, 7, 10, 7, 10, 11, 12, 18, 16, 18, 17, 12, 3]