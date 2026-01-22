import os
from cupy.testing._pytest_impl import is_available, check_available
def multi_gpu(gpu_num):
    """Decorator to indicate number of GPUs required to run the test.

    Tests can be annotated with this decorator (e.g., ``@multi_gpu(2)``) to
    declare number of GPUs required to run. When running tests, if
    ``CUPY_TEST_GPU_LIMIT`` environment variable is set to value greater
    than or equals to 0, test cases that require GPUs more than the limit will
    be skipped.
    """
    check_available('multi_gpu attribute')
    assert 1 < gpu_num

    def _wrapper(f):
        return pytest.mark.skipif(0 <= _gpu_limit < gpu_num, reason='{} GPUs required'.format(gpu_num))(pytest.mark.multi_gpu(f))
    return _wrapper