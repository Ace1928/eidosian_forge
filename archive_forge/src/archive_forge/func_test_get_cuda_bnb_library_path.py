import pytest
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs
def test_get_cuda_bnb_library_path(monkeypatch, cuda120_spec):
    monkeypatch.delenv('BNB_CUDA_VERSION', raising=False)
    assert get_cuda_bnb_library_path(cuda120_spec).stem == 'libbitsandbytes_cuda120'