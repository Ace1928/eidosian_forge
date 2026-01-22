import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.link import (
Test that interval with low > high raises ValueError.