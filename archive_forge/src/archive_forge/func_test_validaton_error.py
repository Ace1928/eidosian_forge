import pytest
from traitlets import HasTraits, TraitError
from ..traittypes import SciType
def test_validaton_error():

    def maxlen(trait, value):
        if len(value) > 10:
            raise ValueError('Too long sequence!')
        return value

    class Foo(HasTraits):
        bar = SciType().valid(maxlen)
    foo = Foo(bar=list(range(5)))
    assert foo.bar == list(range(5))
    with pytest.raises(TraitError):
        foo.bar = list(range(10, 40))
    assert foo.bar == list(range(5))
    foo = Foo(bar=list(range(5, 10)))
    assert foo.bar == list(range(5, 10))