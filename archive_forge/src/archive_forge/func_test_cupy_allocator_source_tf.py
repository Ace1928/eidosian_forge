import pytest
from thinc.util import has_cupy_gpu, has_torch, has_tensorflow, require_gpu
from thinc.backends import use_pytorch_for_gpu_memory, use_tensorflow_for_gpu_memory
from spacy_loggers.cupy import cupy_logger_v1
@pytest.mark.skipif(not has_cupy_gpu or not has_tensorflow, reason='CuPy/TensorFlow support required')
def test_cupy_allocator_source_tf(logger):
    require_gpu()
    use_tensorflow_for_gpu_memory()
    info = {}
    logger(info)
    assert info['cupy.pool.source'] == 'tensorflow'