from typing import Callable
import numpy as np
def left_sample(continuous_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    """Left sample a continuous function.

    Args:
        continuous_pulse: Continuous pulse function to sample.
        duration: Duration to sample for.
        *args: Continuous pulse function args.
        **kwargs: Continuous pulse function kwargs.
    """
    times = np.arange(duration)
    return continuous_pulse(times, *args, **kwargs)