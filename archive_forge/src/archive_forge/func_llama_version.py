import pytest
from transformers import AutoTokenizer
@pytest.fixture(params=LLAMA_VERSIONS)
def llama_version(request):
    return request.param