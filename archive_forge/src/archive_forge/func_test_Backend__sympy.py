from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires('sympy')
def test_Backend__sympy():
    b = Backend('sympy')
    b.sin(b.pi) == 0
    with pytest.raises(AttributeError):
        b.min