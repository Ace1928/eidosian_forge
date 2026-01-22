import pytest  # type: ignore
from google.auth import exceptions  # type:ignore
@pytest.mark.parametrize('retryable', [True, False])
def test_retryable_exceptions(retryable_exception, retryable):
    retryable_exception = retryable_exception(retryable=retryable)
    assert retryable_exception.retryable == retryable