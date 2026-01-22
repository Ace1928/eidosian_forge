from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.parametrize('value_type', numerical_arrow_types)
def test_fsl_to_fsl_cast(value_type):
    cast_type = pa.list_(pa.field('element', value_type), 2)
    dtype = pa.int32()
    type = pa.list_(pa.field('values', dtype), 2)
    fsl = pa.FixedSizeListArray.from_arrays(pa.array([1, 2, 3, 4, 5, 6], type=dtype), type=type)
    assert cast_type == fsl.cast(cast_type).type
    fsl = pa.FixedSizeListArray.from_arrays(pa.array([1, None, None, 4, 5, 6], type=dtype), type=type)
    assert cast_type == fsl.cast(cast_type).type
    dtype = pa.null()
    type = pa.list_(pa.field('values', dtype), 2)
    fsl = pa.FixedSizeListArray.from_arrays(pa.array([None, None, None, None, None, None], type=dtype), type=type)
    assert cast_type == fsl.cast(cast_type).type
    cast_type = pa.list_(pa.field('element', value_type), 3)
    err_msg = 'Size of FixedSizeList is not the same.'
    with pytest.raises(pa.lib.ArrowTypeError, match=err_msg):
        fsl.cast(cast_type)