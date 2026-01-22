import os
import random
import shutil
import tempfile
from functools import partial
import numpy as np
import pytest
from sklearn.datasets import fetch_lfw_pairs, fetch_lfw_people
from sklearn.datasets.tests.test_common import check_return_X_y
from sklearn.utils._testing import assert_array_equal
Check that we properly crop the images.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24942
    