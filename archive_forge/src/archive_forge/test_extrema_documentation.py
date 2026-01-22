import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
Test output for arrays with dimension smaller 3.

        If any dimension of an array is smaller than 3 and `allow_borders` is
        false a footprint, which has at least 3 elements in each
        dimension, can't be applied. This is an implementation detail so
        `local_maxima` should still return valid output (see gh-3261).

        If `allow_borders` is true the array is padded internally and there is
        no problem.
        