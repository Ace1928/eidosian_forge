import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import sctypes, type_info
from ..testing import suppress_warnings
from ..volumeutils import apply_read_scaling, array_from_file, array_to_file, finite_range
from .test_volumeutils import _calculate_scale
Test for scaling / rounding in volumeutils module