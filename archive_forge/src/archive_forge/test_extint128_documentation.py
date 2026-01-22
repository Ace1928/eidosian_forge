import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal

    Iterate over Cartesian product of *args, and if an exception is raised,
    add information of the current iterate.
    