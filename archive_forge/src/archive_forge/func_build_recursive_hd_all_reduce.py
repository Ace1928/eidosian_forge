import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_recursive_hd_all_reduce(input_tensors, red_op, un_op=None):
    """Construct a subgraph for recursive halving-doubling all-reduce.

  The recursive halving-doubling algorithm is described in
  (Thakur et al., 2015).

  The concept is to arrange the participating n devices in
  a linear sequence where devices exchange data pairwise
  with one other device in each round.  During the gather
  phase there are lg(n) rounds where devices exchange
  increasingly smaller sub-tensors with another device
  at increasingly greater distances, until at the top
  each device has 1/n of the fully reduced values.  During the
  scatter phase each device exchanges its fully reduced
  sub-tensor (which doubles in length at each round)
  with one other device at increasingly smaller distances
  until each device has all of the fully reduced values.

  Note: this preliminary version requires that len(input_tensors) be a
    power of 2.  TODO(tucker): relax this restriction.  Also, the
    number of elements in each tensor must be divisible by 2^h where h
    is the number of hops in each phase.  This will also be relaxed in
    the future with edge-case specific logic.

  Args:
    input_tensors: list of `tf.Tensor` to be elementwise reduced.
    red_op: a binary elementwise reduction Op.
    un_op: an optional unary elementwise Op to apply to reduced values.

  Returns:
    list of `tf.Tensor` which are the fully reduced tensors, one
    at each device of input_tensors.

  Raises:
    ValueError: num_devices not a power of 2, or tensor len not divisible
    by 2 the proper number of times.

  References:
    Optimization of Collective Communication Operations in MPICH:
      [Thakur et al., 2005]
      (https://journals.sagepub.com/doi/abs/10.1177/1094342005051521)
      ([pdf](http://wwwi10.lrr.in.tum.de/~gerndt/home/Teaching/HPCSeminar/mpich_multi_coll.pdf))
  """
    devices = [t.device for t in input_tensors]
    input_tensors, shape = _flatten_tensors(input_tensors)
    reduced_shards = _build_recursive_hd_gather(input_tensors, devices, red_op)
    if un_op:
        reduced_shards = [un_op(t) for t in reduced_shards]
    output_tensors = _build_recursive_hd_scatter(reduced_shards, devices)
    if len(shape) != 1:
        output_tensors = _reshape_tensors(output_tensors, shape)
    return output_tensors