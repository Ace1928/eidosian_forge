import numpy as np
from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5

            Evaluates 'simple' tight-frame kernel.

            * simple tf wavelet kernel: supported on [1/4, 1]
            * simple tf scaling function kernel: supported on [0, 1/2]

            Parameters
            ----------
            x : ndarray
                Array of independent variable values
            kerneltype : str
                Can be either 'sf' or 'wavelet'

            Returns
            -------
            r : ndarray

            