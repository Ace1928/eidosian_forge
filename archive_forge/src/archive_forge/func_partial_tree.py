import pytest
from spacy.pipeline._parser_internals import nonproj
from spacy.pipeline._parser_internals.nonproj import (
from spacy.tokens import Doc
@pytest.fixture
def partial_tree():
    return [1, 2, 2, 4, 5, None, 7, 4, 2]