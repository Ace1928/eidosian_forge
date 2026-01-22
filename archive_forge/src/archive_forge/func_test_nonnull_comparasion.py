from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_nonnull_comparasion():
    nonnull1 = NonNull(String)
    nonnull2 = NonNull(String)
    nonnull3 = NonNull(None)
    nonnull1_argskwargs = NonNull(String, None, b=True)
    nonnull2_argskwargs = NonNull(String, None, b=True)
    assert nonnull1 == nonnull2
    assert nonnull1 != nonnull3
    assert nonnull1_argskwargs == nonnull2_argskwargs
    assert nonnull1 != nonnull1_argskwargs