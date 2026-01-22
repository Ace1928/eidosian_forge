from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
import numpy as np
def generalized_discount_cumsum(x: np.ndarray, deltas: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the 'time-dependent' discounted cumulative sum over a
    (reward) sequence `x`.

    Recursive equations:

    y[t] - gamma**deltas[t+1]*y[t+1] = x[t]

    reversed(y)[t] - gamma**reversed(deltas)[t-1]*reversed(y)[t-1] =
    reversed(x)[t]

    Args:
        x (np.ndarray): A sequence of rewards or one-step TD residuals.
        deltas (np.ndarray): A sequence of time step deltas (length of time
            steps).
        gamma: The discount factor gamma.

    Returns:
        np.ndarray: The sequence containing the 'time-dependent' discounted
            cumulative sums for each individual element in `x` till the end of
            the trajectory.

    .. testcode::
        :skipif: True

        x = np.array([0.0, 1.0, 2.0, 3.0])
        deltas = np.array([1.0, 4.0, 15.0])
        gamma = 0.9
        generalized_discount_cumsum(x, deltas, gamma)

    .. testoutput::

        array([0.0 + 0.9^1.0*1.0 + 0.9^4.0*2.0 + 0.9^15.0*3.0,
               1.0 + 0.9^4.0*2.0 + 0.9^15.0*3.0,
               2.0 + 0.9^15.0*3.0,
               3.0])
    """
    reversed_x = x[::-1]
    reversed_deltas = deltas[::-1]
    reversed_y = np.empty_like(x)
    reversed_y[0] = reversed_x[0]
    for i in range(1, x.size):
        reversed_y[i] = reversed_x[i] + gamma ** reversed_deltas[i - 1] * reversed_y[i - 1]
    return reversed_y[::-1]