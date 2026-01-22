import sys
import unittest
from numba import njit

        https://github.com/numba/numba/issues/3027
        Older versions of colorama break stdout/err when recursive functions
        are compiled.

        This test should work irrespective of colorama version, or indeed its
        presence. If the version is too low, it should be disabled and the test
        should work anyway, if it is a sufficiently high version or it is not
        present, it should work anyway.
        