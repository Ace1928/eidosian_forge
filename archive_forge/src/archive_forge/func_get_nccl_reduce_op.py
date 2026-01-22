import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_nccl_reduce_op(reduce_op):
    """Map the reduce op to NCCL reduce op type.

    Args:
        reduce_op: ReduceOp Enum (SUM/PRODUCT/MIN/MAX).
    Returns:
        (nccl.ncclRedOp_t): the mapped NCCL reduce op.
    """
    if reduce_op not in NCCL_REDUCE_OP_MAP:
        raise RuntimeError("NCCL does not support reduce op: '{}'.".format(reduce_op))
    return NCCL_REDUCE_OP_MAP[reduce_op]