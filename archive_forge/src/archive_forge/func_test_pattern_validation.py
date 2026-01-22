import pytest
from spacy.errors import MatchPatternError
from spacy.matcher import Matcher
from spacy.schemas import validate_token_pattern
@pytest.mark.parametrize('pattern,n_errors,_', TEST_PATTERNS)
def test_pattern_validation(pattern, n_errors, _):
    errors = validate_token_pattern(pattern)
    assert len(errors) == n_errors