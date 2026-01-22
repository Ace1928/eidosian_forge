import pytest
from pandas.errors import (
import pandas as pd
def test_AbstractMethodError_classmethod():
    xpr = 'This classmethod must be defined in the concrete class Foo'
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo.classmethod()
    xpr = 'This property must be defined in the concrete class Foo'
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().property
    xpr = 'This method must be defined in the concrete class Foo'
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().method()