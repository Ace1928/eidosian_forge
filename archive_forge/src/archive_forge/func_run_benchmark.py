import time
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.platform import test
def run_benchmark(self, dataset, num_elements, iters=1, warmup=True, apply_default_optimizations=False, session_config=None):
    """Benchmarks the dataset.

    Runs the dataset `iters` times. In each iteration, the benchmark measures
    the time it takes to go through `num_elements` elements of the dataset.

    Args:
      dataset: Dataset to benchmark.
      num_elements: Number of dataset elements to iterate through each benchmark
        iteration.
      iters: Number of times to repeat the timing.
      warmup: If true, warms up the session caches by running an untimed run.
      apply_default_optimizations: Determines whether default optimizations
        should be applied.
      session_config: A ConfigProto protocol buffer with configuration options
        for the session. Applicable only for benchmarking in graph mode.

    Returns:
      A float, representing the per-element wall time of the dataset in seconds.
      This is the median time (with respect to `iters`) it takes for the dataset
      to go through `num_elements` elements, divided by `num_elements.`
    """
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = apply_default_optimizations
    dataset = dataset.with_options(options)
    dataset = dataset.skip(num_elements - 1)
    if context.executing_eagerly():
        median_duration = self._run_eager_benchmark(iterable=dataset, iters=iters, warmup=warmup)
        return median_duration / float(num_elements)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()
    op = nest.flatten(next_element)[0].op
    median_duration = self._run_graph_benchmark(iterable=op, iters=iters, warmup=warmup, session_config=session_config, initializer=iterator.initializer)
    return median_duration / float(num_elements)