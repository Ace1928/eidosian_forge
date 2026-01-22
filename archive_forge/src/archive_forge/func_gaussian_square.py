from __future__ import annotations
import functools
import numpy as np
from qiskit.pulse.exceptions import PulseError
def gaussian_square(times: np.ndarray, amp: complex, center: float, square_width: float, sigma: float, zeroed_width: float | None=None) -> np.ndarray:
    """Continuous gaussian square pulse.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude.
        center: Center of the square pulse component.
        square_width: Width of the square pulse component.
        sigma: Standard deviation of Gaussian rise/fall portion of the pulse.
        zeroed_width: Subtract baseline of gaussian square pulse
            to enforce $\\OmegaSquare(center \\pm zeroed_width/2)=0$.

    Raises:
        PulseError: if zeroed_width is not compatible with square_width.
    """
    square_start = center - square_width / 2
    square_stop = center + square_width / 2
    if zeroed_width:
        if zeroed_width < square_width:
            raise PulseError('zeroed_width cannot be smaller than square_width.')
        gaussian_zeroed_width = zeroed_width - square_width
    else:
        gaussian_zeroed_width = None
    funclist = [functools.partial(gaussian, amp=amp, center=square_start, sigma=sigma, zeroed_width=gaussian_zeroed_width, rescale_amp=True), functools.partial(gaussian, amp=amp, center=square_stop, sigma=sigma, zeroed_width=gaussian_zeroed_width, rescale_amp=True), functools.partial(constant, amp=amp)]
    condlist = [times <= square_start, times >= square_stop]
    return np.piecewise(times.astype(np.complex128), condlist, funclist)