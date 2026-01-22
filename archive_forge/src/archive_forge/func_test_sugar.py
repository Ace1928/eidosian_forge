import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_sugar():
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes)
    assert rc.code == rc.field1
    rc = Recoder(codes, fields=('code1', 'label'))
    assert rc.code1 == rc.field1
    assert rc[1] == rc.field1[1]
    assert rc['two'] == rc.field1['two']
    assert set(rc.keys()) == {1, 'one', '1', 'first', 2, 'two'}
    assert rc.value_set() == {1, 2}
    assert rc.value_set('label') == {'one', 'two'}
    assert 'one' in rc
    assert 'three' not in rc