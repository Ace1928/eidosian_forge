import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
FeatureHasher raises error when a sample is a single string.

    Non-regression test for gh-13199.
    