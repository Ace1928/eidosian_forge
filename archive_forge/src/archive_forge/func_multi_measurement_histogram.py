import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def multi_measurement_histogram(self, *, keys: Iterable[TMeasurementKey], fold_func: Callable[[Tuple], T]=cast(Callable[[Tuple], T], _tuple_of_big_endian_int)) -> collections.Counter:
    """Counts the number of times combined measurement results occurred.

        This is a more general version of the 'histogram' method. Instead of
        only counting how often results occurred for one specific measurement,
        this method tensors multiple measurement results together and counts
        how often the combined results occurred.

        For example, suppose that:

            - fold_func is not specified
            - keys=['abc', 'd']
            - the measurement with key 'abc' measures qubits a, b, and c.
            - the measurement with key 'd' measures qubit d.
            - the circuit was sampled 3 times.
            - the sampled measurement values were:
                1. a=1 b=0 c=0 d=0
                2. a=0 b=1 c=0 d=1
                3. a=1 b=0 c=0 d=0

        Then the counter returned by this method will be:

            collections.Counter({
                (0b100, 0): 2,
                (0b010, 1): 1
            })


        Where '0b100' is binary for '4' and '0b010' is binary for '2'. Notice
        that the bits are combined in a big-endian way by default, with the
        first measured qubit determining the highest-value bit.

        Args:
            fold_func: A function used to convert sampled measurement results
                into countable values. The input is a tuple containing the
                list of bits measured by each measurement specified by the
                keys argument. If this argument is not specified, it defaults
                to returning tuples of integers, where each integer is the big
                endian interpretation of the bits a measurement sampled.
            keys: Keys of measurements to include in the histogram.

        Returns:
            A counter indicating how often measurements sampled various
            results.
        """
    fixed_keys = tuple((_key_to_str(key) for key in keys))
    samples: Iterable[Any] = zip(*(self.measurements[sub_key] for sub_key in fixed_keys))
    if len(fixed_keys) == 0:
        samples = [()] * self.repetitions
    c: collections.Counter = collections.Counter()
    for sample in samples:
        c[fold_func(sample)] += 1
    return c