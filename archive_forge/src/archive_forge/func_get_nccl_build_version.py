import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_nccl_build_version():
    return get_build_version()