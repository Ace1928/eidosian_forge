from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import time
import numpy as np
import six
import tensorflow as tf
def parse_iterations_per_loop(iterations_per_loop):
    """Parses the `iterations_per_loop` value.

  The parser expects the value of the `iterations_per_loop` value to be a
  positive integer value with unit:`count` or time-based value `<N><s|m|h>`
  where <N> is any positive integer and `s`, `m`, `h` are unit of time in
  seconds, minutes, hours respectively. Examples of valid values: `3600s`, `60m`
  , `1h`.

  Args:
    iterations_per_loop: Number of iterations or time alloted to spend on per
      device loop.

  Returns:
    A dictionary of `value` and `unit`. The `unit` value can be either a raw
    `count`, or time in `seconds`.
    {
      "value": <positive-integer>,
      "unit": <unit: `count` | `seconds`>
    }
  """
    m = _ITERATIONS_PER_LOOP_VALUE_REGEX.match(str(iterations_per_loop))
    if m is None:
        raise ValueError('Invalid TPUConfig `iterations_per_loop` value. Value must be positive integer value or time-based value `<N><s|m|h>` where <N> is anypositive integer and `s`, `m`, `h` are unit of time in seconds, minutes, hours respectively. Examples of valid values: `3600s`, `60m`, `1h`.')
    unit_value = 'seconds' if m.group('suffix') in ['h', 'm', 's'] else 'count'
    value = int(m.group('value'))
    if m.group('suffix') == 'm':
        value *= 60
    elif m.group('suffix') == 'h':
        value *= 3600
    return IterationsPerLoopCounter(value, unit_value)