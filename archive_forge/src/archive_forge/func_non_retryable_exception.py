import pytest  # type: ignore
from google.auth import exceptions  # type:ignore
@pytest.fixture(params=[exceptions.ClientCertError])
def non_retryable_exception(request):
    return request.param