import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
@pytest.fixture
def person_org_patterns():
    return [{'label': 'PERSON', 'pattern': 'Dina'}, {'label': 'ORG', 'pattern': 'ACME'}, {'label': 'ORG', 'pattern': 'ACM'}]