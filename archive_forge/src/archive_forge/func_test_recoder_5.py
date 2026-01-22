import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_recoder_5():
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes)
    assert rc.code[1] == 1
    assert rc.code['one'] == 1
    assert rc.code['first'] == 1