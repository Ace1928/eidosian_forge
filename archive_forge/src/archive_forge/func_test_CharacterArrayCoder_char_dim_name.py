from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
@pytest.mark.parametrize(['original', 'expected_char_dim_name'], [(Variable(('x',), [b'ab', b'cdef']), 'string4'), (Variable(('x',), [b'ab', b'cdef'], encoding={'char_dim_name': 'foo'}), 'foo')])
def test_CharacterArrayCoder_char_dim_name(original, expected_char_dim_name) -> None:
    coder = strings.CharacterArrayCoder()
    encoded = coder.encode(original)
    roundtripped = coder.decode(encoded)
    assert encoded.dims[-1] == expected_char_dim_name
    assert roundtripped.encoding['char_dim_name'] == expected_char_dim_name
    assert roundtripped.dims[-1] == original.dims[-1]