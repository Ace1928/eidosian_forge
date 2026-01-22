import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
@pytest.mark.parametrize('op', '+ - * @ / // % ** << >> & ^ |'.split())
def test_all_operators(op):
    m1 = Linear()
    m2 = Linear()
    with Model.define_operators({op: lambda a, b: a.name + b.name}):
        if op == '+':
            value = m1 + m2
        else:
            with pytest.raises(TypeError):
                value = m1 + m2
        if op == '-':
            value = m1 - m2
        else:
            with pytest.raises(TypeError):
                value = m1 - m2
        if op == '*':
            value = m1 * m2
        else:
            with pytest.raises(TypeError):
                value = m1 * m2
        if op == '@':
            value = m1.__matmul__(m2)
        else:
            with pytest.raises(TypeError):
                value = m1.__matmul__(m2)
        if op == '/':
            value = m1 / m2
        else:
            with pytest.raises(TypeError):
                value = m1 / m2
        if op == '//':
            value = m1 // m2
        else:
            with pytest.raises(TypeError):
                value = m1 // m2
        if op == '^':
            value = m1 ^ m2
        else:
            with pytest.raises(TypeError):
                value = m1 ^ m2
        if op == '%':
            value = m1 % m2
        else:
            with pytest.raises(TypeError):
                value = m1 % m2
        if op == '**':
            value = m1 ** m2
        else:
            with pytest.raises(TypeError):
                value = m1 ** m2
        if op == '<<':
            value = m1 << m2
        else:
            with pytest.raises(TypeError):
                value = m1 << m2
        if op == '>>':
            value = m1 >> m2
        else:
            with pytest.raises(TypeError):
                value = m1 >> m2
        if op == '&':
            value = m1 & m2
        else:
            with pytest.raises(TypeError):
                value = m1 & m2
        if op == '^':
            value = m1 ^ m2
        else:
            with pytest.raises(TypeError):
                value = m1 ^ m2
        if op == '|':
            value = m1 | m2
        else:
            with pytest.raises(TypeError):
                value = m1 | m2
    assert Model._context_operators.get() == {}