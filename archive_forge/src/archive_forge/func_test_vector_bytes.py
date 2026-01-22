import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_vector_bytes(self):
    l = [b'a', b'b', b'c']
    s = numpy.array(l, dtype='|S1')
    with (robjects.default_converter + rpyn.converter).context() as cv:
        converted = cv.py2rpy(s)
    assert r['mode'](converted)[0] == 'raw'
    assert r['storage.mode'](converted)[0] == 'raw'
    assert bytearray(b''.join(l)) == bytearray(converted)