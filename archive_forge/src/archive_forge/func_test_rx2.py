import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_rx2(self):
    df = robjects.vectors.DataFrame({'A': robjects.vectors.IntVector([1, 2, 3]), 'B': robjects.vectors.IntVector([1, 2, 3])})
    with (robjects.default_converter + rpyn.converter).context() as cv:
        b = df.rx2('B')
    assert tuple((1, 2, 3)) == tuple(b)