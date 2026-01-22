from __future__ import annotations
from typing import (
import pytest
import numpy as np
import numpy.typing as npt
import numpy._typing as _npt
@pytest.mark.parametrize('name,tup', TYPES.items(), ids=TYPES.keys())
def test_get_args(name: type, tup: TypeTup) -> None:
    """Test `typing.get_args`."""
    typ, ref = (tup.typ, tup.args)
    out = get_args(typ)
    assert out == ref