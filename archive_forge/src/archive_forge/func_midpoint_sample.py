from typing import Callable
import numpy as np
def midpoint_sample(continuous_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    """Sampling strategy for decorator.

    Args:
        continuous_pulse: Continuous pulse function to sample.
        duration: Duration to sample for.
        *args: Continuous pulse function args.
        **kwargs: Continuous pulse function kwargs.
    """
    times = np.arange(1 / 2, duration + 1 / 2)
    return continuous_pulse(times, *args, **kwargs)