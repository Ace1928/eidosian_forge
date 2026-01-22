import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_recoder_2():
    codes = ((1,), (2,))
    rc = Recoder(codes, ['code1'])
    with pytest.raises(AttributeError):
        rc.code
    assert rc.code1[1] == 1
    assert rc.code1[2] == 2