from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_nonnull():
    nonnull = NonNull(String)
    assert nonnull.of_type == String
    assert str(nonnull) == 'String!'