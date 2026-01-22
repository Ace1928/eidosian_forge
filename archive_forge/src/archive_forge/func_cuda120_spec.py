import pytest
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs
@pytest.fixture
def cuda120_spec() -> CUDASpecs:
    return CUDASpecs(cuda_version_string='120', highest_compute_capability=(8, 6), cuda_version_tuple=(12, 0))