from math import exp, pi, sin, sqrt, cos, acos
import numpy as np
from ase.data import atomic_numbers
def calc_pattern(self, x=None, mode='XRD', verbose=False):
    """
        Calculate X-ray diffraction pattern or
        small angle X-ray scattering pattern.

        Parameters:

        x: float array
            points where intensity will be calculated.
            XRD - 2theta values, in degrees;
            SAXS - q values in 1/A
            (`q = 2 \\pi \\cdot s = 4 \\pi \\sin( \\theta) / \\lambda`).
            If ``x`` is ``None`` then default values will be used.

        mode: {'XRD', 'SAXS'}
            the mode of calculation: X-ray diffraction (XRD) or
            small-angle scattering (SAXS).

        Returns:
            list of intensities calculated for values given in ``x``.
        """
    self.mode = mode.upper()
    assert mode in ['XRD', 'SAXS']
    result = []
    if mode == 'XRD':
        if x is None:
            self.twotheta_list = np.linspace(15, 55, 100)
        else:
            self.twotheta_list = x
        self.q_list = []
        if verbose:
            print('#2theta\tIntensity')
        for twotheta in self.twotheta_list:
            s = 2 * sin(twotheta * pi / 180 / 2.0) / self.wavelength
            result.append(self.get(s))
            if verbose:
                print('%.3f\t%f' % (twotheta, result[-1]))
    elif mode == 'SAXS':
        if x is None:
            self.twotheta_list = np.logspace(-3, -0.3, 100)
        else:
            self.q_list = x
        self.twotheta_list = []
        if verbose:
            print('#q\tIntensity')
        for q in self.q_list:
            s = q / (2 * pi)
            result.append(self.get(s))
            if verbose:
                print('%.4f\t%f' % (q, result[-1]))
    self.intensity_list = np.array(result)
    return self.intensity_list