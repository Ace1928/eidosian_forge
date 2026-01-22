import gc
from scipy._lib._gcutils import (set_gc_state, gc_state, assert_deallocated,
from numpy.testing import assert_equal
import pytest
 Test for assert_deallocated context manager and gc utilities
