from typing import Callable, List, Optional, Tuple
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
Initializes a PiecewiseSchedule instance.

        Args:
            endpoints: A list of tuples
                `(t, value)` such that the output
                is an interpolation (given by the `interpolation` callable)
                between two values.
                E.g.
                t=400 and endpoints=[(0, 20.0),(500, 30.0)]
                output=20.0 + 0.8 * (30.0 - 20.0) = 28.0
                NOTE: All the values for time must be sorted in an increasing
                order.
            framework: The framework descriptor string, e.g. "tf",
                "torch", or None.
            interpolation: A function that takes the left-value,
                the right-value and an alpha interpolation parameter
                (0.0=only left value, 1.0=only right value), which is the
                fraction of distance from left endpoint to right endpoint.
            outside_value: If t in call to `value` is
                outside of all the intervals in `endpoints` this value is
                returned. If None then an AssertionError is raised when outside
                value is requested.
        